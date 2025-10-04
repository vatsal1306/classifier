import glob
import multiprocessing
import os

import numpy as np
import webdataset
from tqdm import tqdm


def _write_shard(samples_chunk, shard_path):
    """
    Worker function for multiprocessing. Reads a chunk of image paths
    and writes them to a single WebDataset .tar shard.
    This function runs in a separate process.

    Args:
        samples_chunk (list): A list of tuples, where each tuple is
                              (image_path, label, split_name).
        shard_path (str): The file path to write the output .tar shard to.
    """
    with webdataset.TarWriter(shard_path) as sink:
        for image_path, label, split_name, class_name in samples_chunk:
            try:
                # Read the entire image file in binary mode
                with open(image_path, 'rb') as f:
                    image_data = f.read()

                # Get the file extension (e.g., .jpg, .png)
                _, extension = os.path.splitext(image_path)
                extension = extension.lower().replace('.', '')

                if extension not in ['jpg', 'jpeg', 'png']:
                    continue

                # Create a unique key. Now includes class name for clarity.
                base_name = os.path.basename(image_path)
                key = f"{split_name}_{class_name}_{base_name}"

                # The sample dictionary must have a '__key__'
                sample = {
                    "__key__": key,
                    extension: image_data,
                    "cls": str(label).encode('utf-8')
                }
                sink.write(sample)
            except IOError as e:
                print(f"Warning: Could not read file {image_path}, skipping. Error: {e}")
            except Exception as e:
                print(f"An unexpected error occurred with file {image_path}, skipping. Error: {e}")


class BuildData:
    def __init__(self, input_dir, output_dir):
        """
        Initializes the data building process.

        Args:
            input_dir (str): The root directory of the raw image dataset.
            output_dir (str): The directory where the sharded .tar files will be saved.
        """
        if not input_dir or not output_dir:
            raise ValueError("input_dir and output_dir must be provided.")

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.create_dataset_split('train')
        self.create_dataset_split('test')

    def create_dataset_split(self, split_name):
        """
        Processes a whole data split (e.g., 'train') by handling each class
        within it separately to create class-specific shard directories.
        """
        print(f"========== PROCESSING SPLIT: {split_name.upper()} ==========")
        input_split_path = os.path.join(self.input_dir, split_name)
        if not os.path.isdir(input_split_path):
            print(f"Error: Input directory not found at {input_split_path}")
            return

        class_to_label = {'human': 1, 'non_human': 0}

        # *** KEY CHANGE: Iterate over each class and process it fully ***
        for class_name, label in class_to_label.items():
            print(f"\n--- Processing class: '{class_name}' ---")

            class_path = os.path.join(input_split_path, class_name)

            # Find all images for the current class
            paths = glob.glob(os.path.join(class_path, '**', '*.[jJ][pP][gG]'), recursive=True) + \
                    glob.glob(os.path.join(class_path, '**', '*.[pP][nN][gG]'), recursive=True)

            print(f"Found {len(paths)} images in '{class_name}'")
            if not paths:
                print("No images found. Skipping to next class.")
                continue

            # Create list of samples for THIS CLASS only
            class_samples = [(path, label, split_name, class_name) for path in paths]

            # Define the specific output directory for this class's shards
            output_class_dir = os.path.join(self.output_dir, split_name, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            print(f"Shards will be saved in: {output_class_dir}")

            # Parallelize the writing process for the current class
            self.parallelize_writing(class_samples, output_class_dir, class_name)

        print(f"========== FINISHED PROCESSING SPLIT: {split_name.upper()} ==========")

    def parallelize_writing(self, samples, output_dir, class_name):
        """
        Takes a list of samples and writes them to sharded .tar files in parallel.
        """
        num_cores = multiprocessing.cpu_count()
        chunks = np.array_split(samples, num_cores)

        args_list = []
        for i, chunk in enumerate(chunks):
            if chunk.size > 0:
                shard_path = os.path.join(output_dir, f"shard-{i:06d}.tar")
                args_list.append((chunk.tolist(), shard_path))

        if not args_list:
            print("No data to write.")
            return

        print(f"Using {num_cores} cores to write {len(samples)} images to {len(args_list)} shards...")
        with multiprocessing.Pool(processes=num_cores) as pool:
            with tqdm(total=len(args_list), desc=f"Writing '{class_name}' shards") as pbar:
                for _ in pool.starmap(_write_shard, args_list):
                    pbar.update(1)
        print(f"--- Successfully created shards for class '{class_name}' ---")


class VerifyData:
    def __init__(self, dataset_path, output_filename):
        """
        Initializes the data verification process.

        Args:
            dataset_path (str): Path to the .tar file or a directory of shards.
            output_filename (str): The filename to save the verification image as.
        """
        self.dataset_path = dataset_path
        self.output_filename = output_filename
        self.verify_first_sample()

    def verify_first_sample(self):
        """
        Reads the first sample from a WebDataset using a robust pipeline,
        decodes the image, and saves it to disk for verification.
        """
        # Check if the path is a directory (for sharded data) or a single file
        if os.path.isdir(self.dataset_path):
            # Create a glob pattern to find all shards
            url = os.path.join(self.dataset_path, "shard-*.tar")
            print(f"Reading from sharded dataset at: {self.dataset_path}")
        elif os.path.isfile(self.dataset_path):
            url = self.dataset_path
            print(f"Reading from single dataset file: {self.dataset_path}")
        else:
            print(f"Error: Dataset path not found at {self.dataset_path}")
            return

        try:
            # Create a WebDataset object using the robust decoding pipeline.
            # - .decode("pil"): Automatically decodes image data into PIL Images.
            # - .to_tuple(...): Creates tuples of (image, label).
            #
            # THE FIX: We are now explicitly telling to_tuple to look for the
            # malformed keys like "jpg.jpg" for the image and "jpg.cls" for the label.
            dataset = webdataset.WebDataset(url).decode("pil").to_tuple("jpg.jpg;png.png", "jpg.cls;png.cls")

            # Get the first sample from the dataset iterator. It's now a clean tuple.
            print("Attempting to read the first sample...")
            image_pil, label_bytes = next(iter(dataset))
            # The label is still bytes, so we decode it manually
            # label_int = int(label_bytes.decode('utf-8'))
            label_int = label_bytes
            print("Successfully read and decoded the first sample.")

            # --- Verification Step ---
            # 1. The image is already a PIL Image object.
            # 2. The label has been decoded to an integer.

            # 3. Save the decoded image to a file
            image_pil.save(self.output_filename)

            print("\n--- Verification Successful! ---")
            print(f"Decoded Label: {label_int} ({'human' if label_int == 1 else 'non_human'})")
            print(f"Image Mode/Size: {image_pil.mode} / {image_pil.size}")
            print(f"Image successfully decoded and saved to '{self.output_filename}'")
            print("Please open the file to visually confirm it is correct.")

        except StopIteration:
            print("\nError: The dataset appears to be empty. No samples were found.")
        except Exception as e:
            print(f"\nAn error occurred during verification: {e}")
            # print traceback for more details
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    # BuildData(input_dir='data/filtered_data', output_dir='data/processed')
    VerifyData('/vidgen2/vatsal/classifier/data/processed/test/non_human/shard-000003.tar', 'data/processed/first_image.jpg')
