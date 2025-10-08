#!/usr/bin/env python3
"""
Download a file from your personal Google Drive using OAuth credentials.
This script is designed to work with a token.json file generated from an
OAuth client_id and client_secret.

How to find a Drive file ID:
Right-click the file in the Drive web UI, select "Get link", and copy the ID
from the URL. For example, in '.../d/1a2b3c.../view', the ID is '1a2b3c...'.

Usage Examples:
  # Standard download
  python download_from_drive.py --token token.json --file-id YOUR_FILE_ID --output my_downloaded_file.zip

  # Large file download with progress bar
  python download_from_drive.py --token token.json --file-id YOUR_LARGE_FILE_ID --output big_video.mp4 --large
"""
import argparse
import os
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# tqdm is used for the progress bar. Install it with: uv add tqdm
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm library not found. Please install it with 'uv add tqdm' or 'pip install tqdm'")
    sys.exit(1)

SCOPES = ['https://www.googleapis.com/auth/drive.file']


def build_creds_from_token_file(token_file, save_token_path=None):
    """Builds credentials from a token file and refreshes it if necessary."""
    creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    try:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            save_path = save_token_path or token_file
            if save_path:
                with open(save_path, 'w') as f:
                    f.write(creds.to_json())
                print(f"Token refreshed and saved to {save_path}")
    except Exception as e:
        print(f"Warning: token refresh failed: {e}", file=sys.stderr)
    return creds


def build_drive_service(creds):
    """Builds the Drive v3 service object."""
    return build('drive', 'v3', credentials=creds, cache_discovery=False)


def download_file(service, file_id, output_path, is_large=False):
    """
    Downloads a file from Google Drive.
    Uses a chunked, resumable download which is standard for all file sizes
    and especially robust for large files.
    """
    print(f"Attempting to download file ID: {file_id} to {output_path}")

    try:
        # FIX: Added supportsAllDrives=True to find files in "Shared with me"
        file_metadata = service.files().get(
            fileId=file_id,
            fields='name, size',
            supportsAllDrives=True
        ).execute()

        file_name = file_metadata.get('name')
        file_size = int(file_metadata.get('size', 0))
        print(f"Found file: '{file_name}' ({file_size / 1024 / 1024:.2f} MB)")

        # FIX: Added supportsAllDrives=True here as well for the download request
        request = service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True
        )

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'wb') as fh:
            chunk_size = 16 * 1024 * 1024
            downloader = MediaIoBaseDownload(fh, request, chunksize=chunk_size)

            if is_large:
                print("Using large file download mode (with progress bar)...")
                with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, desc=file_name) as pbar:
                    done = False
                    while not done:
                        status, done = downloader.next_chunk()
                        if status:
                            pbar.update(status.resumable_progress - pbar.n)
            else:
                print("Using standard download mode...")
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"  Downloaded {int(status.progress() * 100)}%.")

        print(f"\nDownload complete! File saved to: {output_path}")

    except HttpError as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        if e.resp.status == 404:
            print(
                f"Error: File not found. Check if the file ID '{file_id}' is correct and that you have permission to access it.",
                file=sys.stderr)
        elif e.resp.status == 403:
            print(
                f"Error: Permission denied or Google's security scanner is blocking the download for this shared file. Try the 'Add shortcut' method.",
                file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nA general error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download a file from Google Drive using a token.json.")
    parser.add_argument('--token', required=True, help='Path to token.json credentials file.')
    parser.add_argument('--file-id', required=True, help='The ID of the file to download from Google Drive.')
    parser.add_argument('--output', required=True, help='The local path to save the downloaded file.')
    parser.add_argument('--large', action='store_true',
                        help='Enable optimized download mode for large files (shows a progress bar).')
    parser.add_argument('--save-token',
                        help='If the token is refreshed, save it to this file path instead of the original.')
    args = parser.parse_args()

    creds = build_creds_from_token_file(args.token, save_token_path=args.save_token)
    service = build_drive_service(creds)

    download_file(service, args.file_id, args.output, args.large)


if __name__ == '__main__':
    main()
