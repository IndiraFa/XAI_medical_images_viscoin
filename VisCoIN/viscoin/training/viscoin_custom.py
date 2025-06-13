"""
Script to train a VisCoIn model with custom parameters
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
from viscoin.utils.logging import configure_score_logging
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.utils import save_viscoin
import dnnlib
import legacy
from viscoin.utils.logging import get_logger
from viscoin.datasets.cub import CUB_200_2011
from viscoin.utils.types import Mode


def initialize_models(n_classes: int, classifier_path: str, n_concepts: int, gan_path: str, adapted_gan_path: str, device: str):
    # initialize classifier
    classifier = Classifier(output_classes=n_classes, pretrained=False).to(device)
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    # initialize concept extractor
    concept_extractor = ConceptExtractor(n_concepts=n_concepts).to(device) # ici initialiser avec le bon nombre de concepts n_concepts (K)

    # initialize explainer
    explainer = Explainer(n_concepts=n_concepts, n_classes=n_classes).to(device)

    # initialize GANs
    with dnnlib.util.open_url(gan_path) as f:
        generator_gan = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    viscoin_gan = GeneratorAdapted(z_dim=n_concepts, mapping_kwargs=adapted_gan_config)
    viscoin_gan = viscoin_gan.from_gan(generator_gan)
    viscoin_gan = viscoin_gan.to(device)

    return classifier, concept_extractor, explainer, viscoin_gan, generator_gan
    

def train_viscoin_custom(batch_size: int, n_epochs: int, alpha: float, beta: float, gamma: float, delta: float, classifier,
        concept_extractor,
        explainer,
        viscoin_gan,
        generator_gan,
        train_loader,
        test_loader,
        device: str):
    # training parameters
    params = TrainingParameters()
    
    params.epochs = n_epochs
    
    # generic params
    params.iterations = 100_000 # default is 100_000
    params.learning_rate = 1e-4 # default value is 1e-4
    params.cd_fid_iteration = 100
    params.batch_size = batch_size
    params.gradient_accumulation = 1

    # Loss coefficients
    params.alpha = alpha
    params.beta = beta
    params.gamma = gamma
    params.delta = delta

    # score logging
    configure_score_logging("viscoin_train_256_mapping.log")

    # train VisCoIN
    train_viscoin(
        classifier,
        concept_extractor,
        explainer,
        viscoin_gan,
        generator_gan,
        train_loader,
        test_loader,
        params,
        device,
    )

    save_viscoin(
        classifier,
        concept_extractor,
        explainer,
        viscoin_gan,
        "viscoin_final.pth"
    )
