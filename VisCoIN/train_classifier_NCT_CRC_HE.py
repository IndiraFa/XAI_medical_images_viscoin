"""
Script to train a classifer on the NCT-CRC-HE dataset resized to 256x256,
taking into account the imbalance between classes, based on adapted 
viscoin.training.classifiers

Author: Indira FABRE
"""

import os
import torch
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from collections import Counter
from viscoin.models.classifiers import Classifier
from viscoin.utils.logging import get_logger
from viscoin.testing.classifiers import test_classifier
from viscoin.datasets.custom_local_dataset import CustomImageDataset, load_datasets, create_dataloaders
from viscoin.datasets.transforms import RESNET_TEST_TRANSFORM, RESNET_TRAIN_TRANSFORM # to normalize the way ResNet and viscoin expect
from PIL import Image
from tqdm import tqdm

# Constants
DATASET_PATH = "datasets/NCT-CRC-HE"
RESIZED_PATH = os.path.join(DATASET_PATH, "resized_images_256")
SAVE_PATH = "VisCoIN/checkpoints/classifier_NCT_CRC_HE.pt"
TARGET_SPLITS = ["NCT_CRC_HE_100K", "CRC_VAL_HE_7K"]
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCH = 10
LEARNING_RATE = 0.0001
LOG_FILE_PATH = "VisCoIN/logs/training_classifier_NCT_CRC_HE.log"

# Initialize logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(LOG_FILE_PATH),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# def load_datasets(resized_path: str, target_splits: set) -> tuple:
#     """
#     Load train and test datasets from the resized images directory.

#     Parameters
#     ----------
#     resized_path : str
#         Path to the resized images directory.
#     target_splits : set
#         Set of target splits to load.

#     Returns
#     -------
#     tuple
#         Train and test datasets.
#     """
#     train_split, test_split = target_splits
#     train_dir = os.path.join(resized_path, train_split)
#     test_dir = os.path.join(resized_path, test_split)

#     train_dataset = CustomImageDataset(train_dir, transform=RESNET_TRAIN_TRANSFORM)
#     test_dataset = CustomImageDataset(test_dir, transform=RESNET_TEST_TRANSFORM)

#     return train_dataset, test_dataset

# def create_dataloaders(train_dataset, test_dataset, batch_size: int) -> tuple:
#     """
#     Create DataLoader instances for train and test datasets.

#     Parameters
#     ----------
#     train_dataset : Dataset
#         Training dataset.
#     test_dataset : Dataset
#         Test dataset.
#     batch_size : int
#         Batch size for DataLoader.

#     Returns
#     -------
#     tuple
#         Train and test DataLoader instances.
#     """
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, test_loader

def initialize_model(output_classes: int, device: str) -> torch.nn.Module:
    """
    Initialize the classifier model.

    Parameters
    ----------
    output_classes : int
        Number of output classes.
    device : str
        Device to use for training ('cuda' or 'cpu').

    Returns
    -------
    torch.nn.Module
        Initialized classifier model.
    """
    model = Classifier(output_classes=output_classes, pretrained=True).to(device)
    return model

def train_classifier(
    model: Classifier,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    model_save_path: str,
    epochs: int = 90,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
):
    """Train the classifier model for the NCT-CRC-HE dataset. 
    The best model on testing data is loaded into the classifier instance.

    Note: the losses are averaged over batches.

    Args:
        model: the classifier model to train
        train_loader: the DataLoader containing the training dataset
        test_loader: the DataLoader containing the testing dataset
        device: the device to use for training
        epochs: the number of epochs to train the model
        learning_rate: the learning rate for the optimizer
        weight_decay: the weight decay for the optimizer
    """
    best_accuracy = 0.0
    best_model = model.state_dict()
    logger = get_logger()

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Calculate class weights
    class_counts = Counter(train_loader.dataset.labels)
    class_weights = torch.tensor([1.0 / class_counts[cls] for cls in range(len(class_counts))], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in tqdm(range(1, epochs + 1), "Training epochs"):
        ###########################################################################################
        #                                      TRAINING STEP                                      #
        ###########################################################################################

        model.train()

        # Training metrics for this epoch
        total_correct = 0
        total_loss = 0
        total_samples = 0

        for inputs, targets in train_loader:
            # Move batch to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute logits
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            current_loss = loss.item()

            # Compute loss and backpropagate
            loss.backward()
            optimizer.step()

            # Update training metrics
            preds = outputs.argmax(dim=1, keepdim=True)
            total_correct += preds.eq(targets.view_as(preds)).sum().item()
            total_loss += current_loss
            total_samples += targets.size(0)

        # Append training metrics
        accuracy = total_correct / total_samples
        batch_mean_loss = total_loss / len(train_loader)
        scheduler.step()

        ###########################################################################################
        #                                       TESTING STEP                                      #
        ###########################################################################################

        test_accuracy, test_mean_loss = test_classifier(model, test_loader, device, criterion, False)

        if accuracy > best_accuracy:
            best_model = model.state_dict()
            best_accuracy = accuracy

        # Log the current state of training
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {batch_mean_loss:.4f} - Train Acc: {100 * accuracy:.2f}% - Test Loss: {test_mean_loss:.4f} - Test Acc: {100 * test_accuracy:.2f}%"
        )

    # Load the best model
    print(f"Best test accuracy: {best_accuracy:.4f}")
    model.load_state_dict(best_model)

    # Save the model
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")


def main():
    """
    Main entry point for training the classifier.
    """
    try:
        logger.info("Loading datasets...")
        train_dataset, test_dataset = load_datasets(RESIZED_PATH, TARGET_SPLITS)

        logger.info("Creating DataLoaders...")
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, BATCH_SIZE)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        logger.info("Initializing model...")
        output_classes = len(set(train_dataset.labels))  # Determine the number of output classes
        logger.info(f"Number of classes: {output_classes}")
        model = initialize_model(output_classes, device)

        logger.info("Training model...")
        train_classifier(model, train_loader, test_loader, device, model_save_path=SAVE_PATH, epochs=EPOCH, learning_rate=LEARNING_RATE)

        logger.info("Training completed.")

    
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
