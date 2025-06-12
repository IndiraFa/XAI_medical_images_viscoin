from torch.utils.data import Dataset
import torch
from torchvision import transforms
import numpy as np

class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        """
        Args:
            hf_dataset (datasets.Dataset): A Hugging Face dataset object with 'image' and 'label' fields.
            transform (callable, optional): Optional torchvision transform to apply to each image.
        """
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample['image']
        label = sample['label']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

