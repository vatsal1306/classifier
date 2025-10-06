import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm


def copy_files():
    src_pth = '/vidgen2/vatsal/classifier/data/data/s3_cluster/data/filter_face_dataset/s3_cluster'
    dest_pth_tr = '/vidgen2/vatsal/classifier/data/filtered_data/train/human'
    dest_pth_te = '/vidgen2/vatsal/classifier/data/filtered_data/test/human'

    c = 0
    dirs = os.listdir(src_pth)
    # n in test
    n = 1200
    train_dirs = dirs[n:]
    test_dirs = dirs[:n]

    for d in train_dirs:
        files = os.listdir(os.path.join(src_pth, d))
        for f in files:
            shutil.move(os.path.join(src_pth, d, f), os.path.join(dest_pth_tr, f))
            c += 1

    print(f"Train files: {c}")
    c = 0
    for d in tqdm(test_dirs):
        files = os.listdir(os.path.join(src_pth, d))
        for f in files:
            shutil.move(os.path.join(src_pth, d, f), os.path.join(dest_pth_te, f))
            c += 1

    print(f"Test files: {c}")


def download_dataset():
    import boto3
    import os
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    def download_file_from_r2(s3, bucket_name, file_key, download_path):
        """Download a single file from R2 with safety checks."""
        try:
            file_info = s3.head_object(Bucket=bucket_name, Key=file_key)
            file_size = file_info['ContentLength']

            os.makedirs(os.path.dirname(download_path), exist_ok=True)
            statvfs = os.statvfs(os.path.dirname(download_path))
            free_space = statvfs.f_bavail * statvfs.f_frsize

            if file_size > free_space:
                print(f"❌ Insufficient storage. "
                      f"File size: {file_size / (1024 ** 3):.2f} GB, "
                      f"Free space: {free_space / (1024 ** 3):.2f} GB.")
                return

            # print(f"⬇️ Downloading {file_key} → {download_path}")
            s3.download_file(bucket_name, file_key, download_path)
            # print(f"✅ Downloaded {download_path}")

        except ClientError as e:
            print(f"❌ Client error for {file_key}: {e.response['Error']['Message']}")
        except BotoCoreError as e:
            print(f"❌ BotoCore error for {file_key}: {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error for {file_key}: {str(e)}")

    def list_files(s3, bucket_name, prefix):
        """List all files under a given prefix."""
        files = []
        continuation_token = None

        while True:
            if continuation_token:
                resp = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix,
                    ContinuationToken=continuation_token
                )
            else:
                resp = s3.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix
                )

            contents = resp.get("Contents", [])
            for item in contents:
                files.append(item["Key"])

            if resp.get("IsTruncated"):  # More files available
                continuation_token = resp["NextContinuationToken"]
            else:
                break

        return files

    def download_directories(access_key, secret_key, endpoint_url, bucket_name, base_dir, dirs):
        """Download all files from the given list of directories in R2."""
        s3 = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint_url,
            region_name='auto',
            config=Config(signature_version="s3v4")
        )

        for d in dirs:
            prefix = f"Web260M/{d}/"
            # print(f"\n📂 Listing files in {prefix}")
            # print(f"Dir {d}/{len(dirs)}")
            keys = list_files(s3, bucket_name, prefix)

            if not keys:
                print(f"⚠️ No files found in {prefix}")
                continue

            for key in tqdm(keys, desc=f"Dir {d}/{len(dirs)}"):
                fname = os.path.basename(key)
                download_path = os.path.join(base_dir, str(d), fname)
                download_file_from_r2(s3, bucket_name, key, download_path)

    download_directories(ACCESS_KEY, SECRET_KEY, R2_ENDPOINT, BUCKET_NAME, BASE_DIR, DIRS)


def copy_one_file(src_path: str, dest_dir: str):
    """
    Copy a single file from src_path into dest_dir.
    Overwrites if already exists.
    """
    # os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.basename(src_path)
    dst = os.path.join(dest_dir, fname)
    try:
        shutil.copy2(src_path, dst)
    except Exception as e:
        # You can log the error or handle partial failures
        print(f"Failed to copy {src_path} to {dst}: {e}")


def gather_files_under(dir_path: str):
    """
    Walk dir_path and return a list of full file paths under it (all subdirectories).
    """
    file_list = []
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            file_list.append(os.path.join(root, f))
    return file_list


def copy_from_sources(src_dirs, target_dir, max_workers, desc):
    """
    For each directory in src_dirs, copy all files (recursively) into target_dir (flat).
    (If you want to preserve subdirectory structure, you can adjust logic.)
    """
    # First, gather all source file paths
    all_files = []
    for sd in src_dirs:
        all_files.extend(gather_files_under(sd))

    total = len(all_files)
    print(f"Copying {total} files into {target_dir} using {max_workers} workers")

    # Use ThreadPoolExecutor for I/O-bound work
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all copy tasks
        futures = [executor.submit(copy_one_file, src, target_dir) for src in all_files]
        # Use tqdm to monitor progress
        for _ in tqdm(as_completed(futures), total=total, desc=desc or f"Copy to {target_dir}"):
            pass  # each iteration means one future finished


def parallel_copy(list_a, list_b, dir_x, dir_y, max_workers):
    """
    Given:
      - list_human: list of source directories whose files go into dir_x
      - list_b: list of source directories whose files go into dir_y
      - dir_x, dir_y: target directories

    Performs both copy operations (A → X, B → Y) in parallel (across threads).
    """
    # You can run the two groups concurrently as well (if desired)
    # But simpler: do group A then group B, both with parallelism

    # print("Copying from list_human into", dir_x)
    copy_from_sources(list_a, dir_x, max_workers=max_workers, desc="From list_a")

    # print("Copying from list_b into", dir_y)
    copy_from_sources(list_b, dir_y, max_workers=max_workers, desc="From list_b")


if __name__ == '__main__':
    # Parameters
    ACCESS_KEY = "04f041b37bfeb0c8cec3aaf1240c23af"
    SECRET_KEY = "e03c7f4047532fb555b9875359f819a1ee8e38c9ce61e141dfcedf2adf6eb636"
    R2_ENDPOINT = "https://afe587cdadc5c79bf6fd36fbfb4e7ac5.r2.cloudflarestorage.com"

    BUCKET_NAME = "datasets"  # only bucket name
    BASE_DIR = '/vidgen2/vatsal/classifier/data/webface'  # where to save

    DIRS = list(
        range(3))  # download 0–9 folders, which means 0 is web4m,  0 to 2 is Web12m and 0 to 9 is web42m(web260m)

    # copy_files()
    # download_dataset()

    list_human = [
        "/vidgen2/vatsal/classifier/dataset/new/data/bm/human",
        '/vidgen2/vatsal/classifier/dataset/new/data/bm/unique_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model/val/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model/train/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model_training/train/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model_training/val/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_scraped_bm_set/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/onnx_bm/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/onnx_output/human',
        '/vidgen2/vatsal/classifier/dataset/new/data/preprocessed_dataset/val/real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/train/real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/benchmark/real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/val/real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/s3_sorted/human',
    ]

    list_non_human = [
        '/vidgen2/vatsal/classifier/dataset/new/data/bm/nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/bm/unique_nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model/val/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model/train/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model_training/train/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_dataset_for_model_training/val/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/new_scraped_bm_set/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/onnx_bm/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/onnx_output/non_human',
        '/vidgen2/vatsal/classifier/dataset/new/data/preprocessed_dataset/train/not_real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/preprocessed_dataset/val/not_real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/train/not_real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/benchmark/not_real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/processed_dataset_sourav/val/not_real_face',
        '/vidgen2/vatsal/classifier/dataset/new/data/s3_sorted/nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/anime_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/cartoon_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/comic_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/full_face_masks',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/game_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/paintings',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/photo_of_nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_madhu_data/statues',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/anime_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/cartoon_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/comic_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/full_face_masks',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/game_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/paintings',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/photo_of_nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_new_scraped/statues',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/anime_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/cartoon_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/comic_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/full_face_masks',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/game_characters',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/paintings',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/photo_of_nonhuman',
        '/vidgen2/vatsal/classifier/dataset/new/data/sorted_scraped/statues',
        '/vidgen2/vatsal/classifier/dataset/new/data/unique_nonhuman',
    ]

    dest_human = "/vidgen2/vatsal/classifier/dataset/processed/train/human"
    dest_non_human = "/vidgen2/vatsal/classifier/dataset/processed/train/non_human"

    # You could set max_workers = os.cpu_count() * 2 or some fixed number
    parallel_copy(list_human, list_non_human, dest_human, dest_non_human, max_workers=os.cpu_count() * 2)
