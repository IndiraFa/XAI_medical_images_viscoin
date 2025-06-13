"""
Script to train a VisCoIN on the NCT-CRC-HE dataset resized to 256x256,
adapted from https://github.com/GnRlLeclerc/VisCoIN

Author: Indira FABRE
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
from viscoin.models.utils import load_viscoin
from viscoin.training.viscoin import train_viscoin, TrainingParameters
from viscoin.training.viscoin_custom import initialize_models, train_viscoin_custom
from viscoin.utils.logging import configure_score_logging
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.utils import save_viscoin
from viscoin.datasets.custom_local_dataset import load_datasets, create_dataloaders
import dnnlib
import legacy
from viscoin.utils.logging import get_logger

logger = get_logger()

# Constants
DATASET_PATH = "datasets/NCT-CRC-HE"
RESIZED_PATH = os.path.join(DATASET_PATH, "resized_images_256")
CLASSIFIER_PATH = "VisCoIN/checkpoints/classifer_model.pt"
GAN_PATH = "VisCoIN/stylegan2_ada/training-runs/00000-NCT_CRC_HE_100K-auto1-kimg1000-resumecustom/network-snapshot-001000.pkl" # à adapter
ADAPTED_GAN_PATH = "VisCoIN/checkpoints/adapted_gan_nct_crc_he_256/gan_adapted256.pth"
TARGET_SPLITS = ["NCT_CRC_HE_100K", "CRC_VAL_HE_7K"]
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8 # same as reference paper
EPOCH = 10
N_CONCEPTS=256
NAME = 'defaultparams'

# Loss coefficients
ALPHA = 0.5  # Output fidelity loss
BETA = 3.0  # LPIPS loss
GAMMA = 0.1  # Reconstruction classification loss, default is 0.1
DELTA = 0.2  # Sparsity loss default is 0.2

adapted_gan_config = {
    "coarse_layer": 4,
    "mid_layer": 12,
    "num_layers": 1,
}

# def initialize_models(n_classes: int, classifier_path: str, n_concepts: int, gan_path: str, adapted_gan_path: str, device: str):
#     # initialize classifier
#     classifier = Classifier(output_classes=n_classes, pretrained=False).to(device)
#     classifier.load_state_dict(torch.load(classifier_path, map_location=device))

#     # initialize concept extractor
#     concept_extractor = ConceptExtractor(n_concepts=n_concepts).to(device) # ici initialiser avec le bon nombre de concepts n_concepts (K)

#     # initialize explainer
#     explainer = Explainer(n_concepts=n_concepts, n_classes=n_classes).to(device)

#     # initialize GANs
#     with dnnlib.util.open_url(gan_path) as f:
#         generator_gan = legacy.load_network_pkl(f)['G_ema'].to(device)
        
#     viscoin_gan = GeneratorAdapted(z_dim=n_concepts, mapping_kwargs=adapted_gan_config)
#     viscoin_gan = viscoin_gan.from_gan(generator_gan)
#     viscoin_gan = viscoin_gan.to(device)

#     return classifier, concept_extractor, explainer, viscoin_gan, generator_gan
    

# def train_viscoin_custom(batch_size: int, n_epochs: int, alpha: float, beta: float, gamma: float, delta: float, classifier,
#         concept_extractor,
#         explainer,
#         viscoin_gan,
#         generator_gan,
#         train_loader,
#         test_loader,
#         device: str, 
#         name: str):
#     # training parameters
#     params = TrainingParameters()
    
#     params.epochs = n_epochs
    
#     # generic params
#     params.iterations = 100_000 # default is 100_000
#     params.learning_rate = 1e-4 # default value is 1e-4
#     params.cd_fid_iteration = 100
#     params.batch_size = batch_size
#     params.gradient_accumulation = 1

#     # Loss coefficients
#     params.alpha = alpha
#     params.beta = beta
#     params.gamma = gamma
#     params.delta = delta

#     # score logging
#     configure_score_logging("viscoin_train_256_mapping_defaultparams.log")

#     # train VisCoIN
#     train_viscoin(
#         classifier,
#         concept_extractor,
#         explainer,
#         viscoin_gan,
#         generator_gan,
#         train_loader,
#         test_loader,
#         params,
#         device,
#         name
#     )

#     save_viscoin(
#         classifier,
#         concept_extractor,
#         explainer,
#         viscoin_gan,
#         "viscoin_final.pth"
#     )


def main():
    """
    Main entry point for training the viscoin model.
    """
    try:
        logger.info("Loading datasets...")
        train_dataset, test_dataset = load_datasets(RESIZED_PATH, TARGET_SPLITS)

        logger.info("Creating DataLoaders...")
        train_loader, test_loader = create_dataloaders(train_dataset, test_dataset, BATCH_SIZE)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")

        logger.info("Initializing model...")
        n_classes = len(set(train_dataset.labels))  # Determine the number of output classes
        classifier, concept_extractor, explainer, viscoin_gan, generator_gan = initialize_models(n_classes, classifier_path=CLASSIFIER_PATH, n_concepts=N_CONCEPTS, gan_path=GAN_PATH, adapted_gan_path=ADAPTED_GAN_PATH, device=device)

        logger.info("Training model...")
        train_viscoin_custom(batch_size=BATCH_SIZE, n_epochs=EPOCH, alpha=ALPHA, beta=BETA, gamma=GAMMA, delta=DELTA, classifier=classifier, concept_extractor=concept_extractor, explainer=explainer, viscoin_gan=viscoin_gan, generator_gan=generator_gan, train_loader=train_loader, test_loader=test_loader, device=device, name=NAME)

        logger.info("Training completed.")

    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    

if __name__ == "__main__":
    main()
