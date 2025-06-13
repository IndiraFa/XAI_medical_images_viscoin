"""
Script to download the NCT-CRC-HE dataset, save it to disk (if needed),
and resize selected splits to 256x256 PNG images organized by label.

Author: Indira FABRE
"""

from datasets import load_dataset, DatasetDict
from PIL import Image
from tqdm import tqdm
import os

# Constants
DATASET_NAME = "1aurent/NCT-CRC-HE"
DATASET_PATH = "."
RESIZED_PATH = os.path.join(DATASET_PATH, "resized_images_256")
TARGET_SPLITS = {"NCT_CRC_HE_100K", "CRC_VAL_HE_7K"}
IMAGE_SIZE = (256, 256)

def load_dataset_optional_save(dataset_name: str, save_path: str = None, save_original: bool = False) -> DatasetDict:
    """
    Load the dataset from HuggingFace hub. Optionally save it to disk.

    Parameters
    ----------
    dataset_name : str
        Hugging Face dataset name (e.g., "1aurent/NCT-CRC-HE").
    save_path : str, optional
        Path to save the dataset if `save_original` is True.
    save_original : bool
        Whether to save the downloaded dataset to disk.

    Returns
    -------
    DatasetDict
        The loaded dataset.
    """
    dataset = load_dataset(dataset_name)

    if save_original and save_path:
        os.makedirs(save_path, exist_ok=True)
        print(f"Saving dataset to {save_path}...")
        dataset.save_to_disk(save_path)

    return dataset

def resize_and_save_images(dataset: DatasetDict, output_dir: str, image_size=(256, 256)):
    """
    Resize images in selected dataset splits and save them as PNGs grouped by label.

    Parameters
    ----------
    dataset : DatasetDict
        A HuggingFace `DatasetDict` containing image splits.
    output_dir : str
        Root directory to save the resized images.
    image_size : tuple of int, optional
        Target size (width, height) for resized images. Default is (256, 256).
    """
    # Ensure the root output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in dataset.items():
        if split_name not in TARGET_SPLITS:
            continue

        print(f"Processing split: {split_name}...")
        for i, example in enumerate(tqdm(split_data, desc=f"Resizing {split_name}")):
            label = str(example["label"])
            label_dir = os.path.join(output_dir, split_name, label)
            os.makedirs(label_dir, exist_ok=True)

            image = example["image"].resize(image_size, Image.Resampling.LANCZOS)
            image_path = os.path.join(label_dir, f"image_{i}.png")
            image.save(image_path)

def main():
    """
    Main entry point for dataset processing:
    - Loads dataset (from disk or HuggingFace hub)
    - Resizes images from selected splits
    - Saves them to disk organized by label
    """
    dataset = load_dataset_optional_save(DATASET_NAME, save_path=DATASET_PATH, save_original=False)
    resize_and_save_images(dataset, RESIZED_PATH, IMAGE_SIZE)

if __name__ == "__main__":
    main()
