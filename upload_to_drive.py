"""
upload a single file, multiple files, a whole directory (recursively) or multiple directories to your personal Google Drive
using your OAuth client_id, client_secret and refresh_token.

How to find a Drive folder ID:
Open the folder in the Drive web UI and look in the URL — the long id after /folders/ is the folder ID

Install dependencies:
pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib

Usage examples:
  # upload a directory (mirrors local folder structure into Drive)
  python upload_to_drive.py --token token.json --dir /path/to/myfolder --parent DRIVE_PARENT_ID

  # upload multiple directories
  python upload_to_drive.py --token token.json --dirs /path/to/dirA /path/to/dirB /path/to/dirC --parent DRIVE_PARENT_ID

  # multiple files
  python upload_to_drive.py --token token.json --files /a/x.png /b/y.jpg --parent DRIVE_PARENT_ID

"""
import argparse
import json
import mimetypes
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

TOKEN_URI = 'https://oauth2.googleapis.com/token'
SCOPES = ['https://www.googleapis.com/auth/drive.file']  # least-privilege for uploads


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_creds_from_auth(auth_json, save_token_path=None):
    """
    auth_json: dict with keys client_id, client_secret, refresh_token (access_token optional)
    If refresh_token present, try to refresh immediately to get a fresh access token.
    If save_token_path provided, updated credentials are saved there in Google token format.
    """
    client_id = auth_json.get('client_id')
    client_secret = auth_json.get('client_secret')
    access_token = auth_json.get('access_token')  # may be None
    refresh_token = auth_json.get('refresh_token')

    creds = Credentials(
        token=access_token,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES
    )

    # Try to refresh immediately if refresh_token available
    try:
        if creds.refresh_token:
            creds.refresh(Request())
            if save_token_path:
                with open(save_token_path, 'w') as f:
                    f.write(creds.to_json())
    except Exception as e:
        # don't crash — we may still have a valid access_token already
        print("Warning: token refresh failed (will continue if access_token present):", e)

    return creds


def build_creds_from_token_file(token_file, save_token_path=None):
    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    # try refresh if possible and persist
    try:
        if creds.refresh_token:
            creds.refresh(Request())
            if save_token_path:
                with open(save_token_path, 'w') as f:
                    f.write(creds.to_json())
    except Exception as e:
        print("Warning: token refresh failed:", e)
    return creds


def build_drive_service(creds):
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def find_folder(service, name, parent_id=None):
    """Return folder id if exists (first match), else None."""
    q = "mimeType='application/vnd.google-apps.folder' and name = '{}'".format(name.replace("'", "\\'"))
    if parent_id:
        q += " and '{}' in parents".format(parent_id)
    else:
        # search in root by default
        q += " and 'root' in parents"
    try:
        res = service.files().list(q=q, spaces='drive', fields='files(id,name)', pageSize=10).execute()
        files = res.get('files', [])
        if files:
            return files[0]['id']
    except HttpError as e:
        print("Warning: failed to search for folder:", e)
    return None


def create_folder(service, name, parent_id=None):
    body = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        body['parents'] = [parent_id]
    folder = service.files().create(body=body, fields='id,name').execute()
    return folder.get('id')


def ensure_drive_path(service, base_parent_id, rel_path):
    """
    Ensure a folder path exists under base_parent_id. rel_path is like 'a/b/c' (OS path separators accepted).
    Returns the id of the deepest folder.
    """
    parts = [p for p in rel_path.replace("\\", "/").split('/') if p and p != '.']
    parent = base_parent_id
    for part in parts:
        found = find_folder(service, part, parent)
        if found:
            parent = found
        else:
            parent = create_folder(service, part, parent)
    return parent


def upload_file(service, local_path, remote_name=None, parent_id=None, chunk_size_mb=8):
    remote_name = remote_name or os.path.basename(local_path)
    mtype, _ = mimetypes.guess_type(local_path)
    media = MediaFileUpload(local_path, mimetype=mtype, resumable=True, chunksize=chunk_size_mb * 1024 * 1024)
    meta = {'name': remote_name}
    if parent_id:
        meta['parents'] = [parent_id]

    req = service.files().create(body=meta, media_body=media, fields='id, name')
    response = None
    print(f"Uploading: {local_path} -> {remote_name} (parent={parent_id})")
    while response is None:
        try:
            status, response = req.next_chunk()
        except HttpError as e:
            print("Upload failed (HttpError):", e)
            raise
        if status:
            pct = int(status.progress() * 100)
            print(f"  {pct}%")
    print("Upload complete. File ID:", response.get('id'))
    return response.get('id')


def upload_multiple_files(service, paths, parent_id=None):
    ids = []
    for p in paths:
        ids.append(upload_file(service, p, parent_id=parent_id))
    return ids


def upload_directory(service, local_dir, parent_id=None):
    local_dir = os.path.abspath(local_dir)
    root_name = os.path.basename(local_dir.rstrip(os.sep))
    # create a root folder for this directory on Drive (under parent_id).
    root_drive_folder = find_folder(service, root_name, parent_id)
    if not root_drive_folder:
        root_drive_folder = create_folder(service, root_name, parent_id)
    print(f"Uploading directory '{local_dir}' into Drive folder id {root_drive_folder}")

    # walk and mirror structure
    for root, dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        if rel == '.':
            drive_parent = root_drive_folder
        else:
            drive_parent = ensure_drive_path(service, root_drive_folder, rel)
        for fname in files:
            local_path = os.path.join(root, fname)
            upload_file(service, local_path, parent_id=drive_parent)
    return root_drive_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--token', help='Path to token.json (google token format produced by InstalledAppFlow)')
    parser.add_argument('--save-token',
                        help='If set, save refreshed token JSON to this file (e.g., updated_token.json)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', help='Upload a single file (path)')
    group.add_argument('--files', nargs='+', help='Upload multiple files')
    group.add_argument('--dir', help='Upload a directory (recursively)')
    group.add_argument('--dirs', nargs='+', help='Upload multiple directories (recursively)')
    parser.add_argument('--parent',
                        help='Drive parent folder ID under which to upload (default: your Drive root)')
    args = parser.parse_args()

    if args.token:
        creds = build_creds_from_token_file(args.token, save_token_path=args.save_token)
    else:
        print("Provide --token")
        sys.exit(1)

    service = build_drive_service(creds)

    try:
        if args.file:
            upload_file(service, args.file, parent_id=args.parent)
        elif args.files:
            upload_multiple_files(service, args.files, parent_id=args.parent)
        elif args.dir:
            upload_directory(service, args.dir, parent_id=args.parent)
        elif args.dirs:
            upload_multiple_directories(service, args.dirs, parent_id=args.parent)

    except HttpError as e:
        print("Drive API returned an error:", e)


if __name__ == '__main__':
    main()
