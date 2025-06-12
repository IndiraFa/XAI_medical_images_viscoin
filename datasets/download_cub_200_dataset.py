"""
Script to resize the CUB dataset, already downloaded.

Author: Indira FABRE
"""

import os
from PIL import Image
from torchvision import transforms

# Define the transformation
TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Constants
INPUT_FOLDER = "./CUB-200/CUB_200_2011"
OUTPUT_FOLDER = "./CUB-200/resized_images_256"

def resize_and_save_images(input_folder, output_folder):
    """
    Resize images from the input folder and its subfolders, and save them to the output folder
    with the same structure.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing the images.
    output_folder : str
        Path to the folder where resized images will be saved.
    """
    # Walk through the input directory and its subdirectories
    for root, dirs, files in os.walk(input_folder):
        # Create the corresponding output directory structure
        relative_path = os.path.relpath(root, input_folder)
        output_dir = os.path.join(output_folder, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # Process each file in the current directory
        for filename in files:
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                # Open the image file
                image_path = os.path.join(root, filename)
                image = Image.open(image_path)

                # Apply the transformations
                transformed_image = TRANSFORM(image)

                # Save the transformed image
                output_path = os.path.join(output_dir, filename)
                transforms.ToPILImage()(transformed_image).save(output_path)

def main():
    """
    Main entry point for image processing:
    - Resizes images from the input folder
    - Saves them to the output folder
    """
    resize_and_save_images(INPUT_FOLDER, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()
