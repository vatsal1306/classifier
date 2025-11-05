import argparse
import math
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from PIL import Image


def make_one_collage(img_paths, imgs_per_row, imgs_per_col, img_size, out_path):
    """Generate one collage from a list of image paths (exactly imgs_per_row * imgs_per_col or fewer)."""
    w, h = img_size
    collage_w = imgs_per_row * w
    collage_h = imgs_per_col * h
    collage = Image.new("RGB", (collage_w, collage_h))
    idx = 0
    for r in range(imgs_per_col):
        for c in range(imgs_per_row):
            if idx >= len(img_paths):
                break
            try:
                im = Image.open(img_paths[idx])
                if im.size != img_size:
                    im = im.resize(img_size)
                collage.paste(im, (c * w, r * h))
            except Exception as e:
                print(f"Error opening/pasting {img_paths[idx]}: {e}")
            idx += 1
        if idx >= len(img_paths):
            break
    collage.save(out_path, quality=90)
    return out_path


def worker_collage_chunk(chunk_idx, chunk_paths, imgs_per_row, imgs_per_col, img_size, out_prefix, out_dir):
    """Worker function: builds collage for a chunk (for multiprocessing)."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_prefix}_collage_{chunk_idx}.jpg")
    return make_one_collage(chunk_paths, imgs_per_row, imgs_per_col, img_size, out_path)


def make_collages_parallel(
        img_dir,
        out_dir,
        out_prefix="collage",
        imgs_per_row=20,
        imgs_per_col=10,
        img_size=(332, 112),
        ext_whitelist={".jpg", ".jpeg", ".png"},
        num_workers=None
):
    """
    Splits all images in img_dir into chunks and parallelizes collage creation.
    """
    # collect image paths
    all_paths = []
    for root, dirs, files in os.walk(img_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in ext_whitelist:
                all_paths.append(os.path.join(root, fname))
    all_paths.sort()
    total = len(all_paths)
    per_collage = imgs_per_row * imgs_per_col
    num_chunks = math.ceil(total / per_collage)

    # partition into chunks
    chunks = []
    for i in range(num_chunks):
        chunk = all_paths[i * per_collage: (i + 1) * per_collage]
        chunks.append((i, chunk))

    # decide num_workers
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    print(f"Total images = {total}, chunk size = {per_collage}, num chunks = {num_chunks}, using {num_workers} workers")

    # partial for worker
    worker = partial(
        worker_collage_chunk,
        imgs_per_row=imgs_per_row,
        imgs_per_col=imgs_per_col,
        img_size=img_size,
        out_prefix=out_prefix,
        out_dir=out_dir
    )

    with Pool(num_workers) as pool:
        results = pool.starmap(worker, chunks)

    print("All done. Collages saved:")
    for r in results:
        print("  ", r)


if __name__ == "__main__":
    # make_collages_parallel(
    #     img_dir="runs/effnet_s_manclean_b1/infer/infer_data/non_human",
    #     out_dir="runs/effnet_s_manclean_b1/infer/grids/infer_data/non_human",
    #     out_prefix="tile",
    #     imgs_per_row=20,
    #     imgs_per_col=10,
    #     # img_size=(332, 112), # for model test pipeline results
    #     img_size=(112, 112),   # for 25K inference data
    #     num_workers=8  # or leave None to auto pick
    # )

    parser = argparse.ArgumentParser(description="Create grids of images.")

    # Required arguments
    parser.add_argument(
        '-i', "--img_dir",
        type=str,
        required=True,
        help="Directory containing the input images."
    )
    parser.add_argument(
        '-o', "--out_dir",
        type=str,
        required=True,
        help="Directory where the output collages will be saved."
    )

    # Optional arguments with defaults
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="tile",
        help="Prefix for the output collage filenames. (Default: tile)"
    )
    parser.add_argument(
        "--imgs_per_row",
        type=int,
        default=20,
        help="Number of images per row in the collage. (Default: 20)"
    )
    parser.add_argument(
        "--imgs_per_col",
        type=int,
        default=10,
        help="Number of images per column in the collage. (Default: 10)"
    )

    # Special handling for the tuple argument
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,  # Expects two values
        default=[112, 112],
        metavar=("WIDTH", "HEIGHT"),
        help="Target size (width height) for each image. (Default: 112 112)"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of worker processes to use. (Default: 8)"
    )

    args = parser.parse_args()

    # Convert the img_size list [W, H] to a tuple (W, H)
    img_size_tuple = tuple(args.img_size)

    # Call the function with the parsed arguments
    make_collages_parallel(
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
        imgs_per_row=args.imgs_per_row,
        imgs_per_col=args.imgs_per_col,
        img_size=img_size_tuple,
        num_workers=args.num_workers
    )
