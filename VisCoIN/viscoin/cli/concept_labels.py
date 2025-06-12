import click
import clip
import numpy as np
import torch

from viscoin.cli.utils import concept2clip_path, device, viscoin_pickle_path, vocab_path
from viscoin.datasets.cub import CUB_200_2011
from viscoin.models.utils import load_viscoin_pickle
from viscoin.testing.concept2clip import get_concept_labels_vocab
from viscoin.testing.concept_label_metric import evaluate_concept_labels


@click.command()
@concept2clip_path
@vocab_path
@viscoin_pickle_path
@click.option(
    "--n-concepts",
    help="The number of concepts",
    type=int,
    default=256,
)
@click.option(
    "--amplify-multiplier",
    help="The multiplier to amplify to the concept",
    type=float,
    default=4.0,
)
@click.option(
    "--selection-n",
    help="The number of images to select for each concept",
    type=int,
    default=100,
)
@click.option(
    "--output_path",
    help="The path to save the concept labels",
    type=str,
    default="concept_labels.csv",
)
@device
def clip_concept_labels(
    concept2clip_path: str,
    vocab_path: str,
    viscoin_pickle_path: str,
    n_concepts: int,
    amplify_multiplier: float,
    selection_n: int,
    output_path: str,
    device: str,
):
    """
    Generate concept labels from a given vocabulary using the clip adapter model :

        We select the most activating images for each concept based on a threshold.
        For each image, we compute its CLIP embedding and the CLIP embedding of the image where the concept is amplified.
        We compute the difference between the two and compute the similarity with the vocabulary.
        We average the similarity for each concept and save the results to a file.

    Args:
        concept2clip_path (str): Path to the clip adapter model
        vocab_path (str): Path to the vocabulary file (.txt)
        viscoin_pickle_path (str): Path to the viscoin pickle file
        n_concepts (int): The number of concepts
        dataset_path (str): Path to the dataset
        amplify_multiplier (float): The multiplier to amplify to the concept
        selection_n (int): The number of images to select for each concept
        output_path (str): The path to save the concept labels
    """

    # Load CLIP adapter model and CLIP model
    concept2clip = torch.load(concept2clip_path, weights_only=False).to(device)

    clip_model, _ = clip.load("ViT-B/16", device=device)

    # Load Viscoin Classifier and Concept Extractor
    viscoin = load_viscoin_pickle(viscoin_pickle_path)
    viscoin.classifier = viscoin.classifier.to(device)
    viscoin.concept_extractor = viscoin.concept_extractor.to(device)

    # Load the vocabulary from which to chose the concept labels
    with open(vocab_path, "r") as f:
        vocab = f.readlines()

    vocab = [v.strip() for v in vocab]

    # Load the dataset
    dataset = CUB_200_2011(mode="test")

    concept_labels, probs, most_activating_images = get_concept_labels_vocab(
        concept2clip,
        viscoin.concept_extractor,
        viscoin.classifier,
        clip_model,
        vocab,
        n_concepts,
        dataset,
        amplify_multiplier,
        selection_n,
        device,
    )

    # Save to file
    with open(output_path, "w") as f:

        f.write("unit,description,similarity,most-activating-images\n")

        for i, label in enumerate(concept_labels):
            f.write(
                f"{i},{label},{probs[i]},{":".join(np.char.mod("%i", most_activating_images[i]))}\n"
            )


@click.command()
@click.option(
    "--expert-annotations-score-path",
    help="The path to the expert annotations score file",
    default="./checkpoints/saved_expert_annotations_score.npy",
    type=str,
    required=True,
)
@viscoin_pickle_path
@click.option(
    "--concept-labels-path",
    help="The path to the predicted concept labels file",
    type=str,
    required=True,
)
@device
@click.option(
    "--evaluation-method",
    help="The evaluation method to use: only topk is available for now",
    type=str,
    default="topk",
)
@click.option(
    "--topk-value",
    help="The value of k to use for the topk evaluation method",
    type=int,
    default=5,
)
@click.option(
    "--neurons-to-study",
    help="The indices of the neurons to study, if empty, 5 random neurons will be selected",
    type=str,
    default="",  # Empty list means random neurons
)
def evalutate_concept_captions(
    expert_annotations_score_path: str,
    viscoin_pickle_path: str,
    concept_labels_path: str,
    device: str,
    evaluation_method: str,
    topk_value: int,
    neurons_to_study: str,
):
    """
    Evaluate the provided predictions of concept labels against cub expert annotations.
    """

    neurons_indexes_to_study = (
        [int(i) for i in neurons_to_study.split(",")] if neurons_to_study else []
    )

    evaluate_concept_labels(
        expert_annotations_score_path,
        viscoin_pickle_path,
        concept_labels_path,
        device,
        evaluation_method,
        topk_value,
        neurons_indexes_to_study,
    )
