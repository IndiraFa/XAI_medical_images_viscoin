"""
Script to train a classifer on the CUB 200 dataset resized to 256x256,
taking into account the imbalance between classes, based on viscoin.training.classifiers

Author: Indira FABRE
"""
import torch
from torch.utils.data import DataLoader
from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.classifiers import Classifier
from viscoin.training.classifiers import train_classifier
from viscoin.utils.types import Mode

# Parameters
NUM_CLASSES = 200
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0001

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    print("Loading CUB-200-2011 dataset...")
    train_dataset = CUB_200_2011(mode="train")
    test_dataset = CUB_200_2011(mode="test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize model
    print("Initializing model...")
    model = Classifier(output_classes=NUM_CLASSES).to(device)

    # Train
    print("Starting training...")
    train_classifier(model, train_loader, test_loader, device=device, epochs=EPOCHS, learning_rate=LEARNING_RATE)

    # Save model
    print("Saving best model...")
    torch.save(model.state_dict(), "cub_classifier.pth")
    print("Done.")

if __name__ == "__main__":
    main()

