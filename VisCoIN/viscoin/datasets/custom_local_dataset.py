"""
Custom class for loading a local dataset of images
and create dataloaders
Author : Indira Fabre
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_images()

    def _load_images(self):
        for label in os.listdir(self.image_dir):
            label_dir = os.path.join(self.image_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for image_name in os.listdir(label_dir):
                if image_name.endswith('.png'):
                    self.image_paths.append(os.path.join(label_dir, image_name))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
    
def load_datasets(resized_path: str, target_splits: set) -> tuple:
    """
    Load train and test datasets from the resized images directory.

    Parameters
    ----------
    resized_path : str
        Path to the resized images directory.
    target_splits : set
        Set of target splits to load.

    Returns
    -------
    tuple
        Train and test datasets.
    """
    train_split, test_split = target_splits
    train_dir = os.path.join(resized_path, train_split)
    test_dir = os.path.join(resized_path, test_split)

    train_dataset = CustomImageDataset(train_dir, transform=RESNET_TRAIN_TRANSFORM)
    test_dataset = CustomImageDataset(test_dir, transform=RESNET_TEST_TRANSFORM)

    return train_dataset, test_dataset

def create_dataloaders(train_dataset, test_dataset, batch_size: int) -> tuple:
    """
    Create DataLoader instances for train and test datasets.

    Parameters
    ----------
    train_dataset : Dataset
        Training dataset.
    test_dataset : Dataset
        Test dataset.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    tuple
        Train and test DataLoader instances.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
