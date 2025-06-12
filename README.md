# Implementation of VisCoIN for the visualisation of concepts activated in the classification of medical images

## Context

This work is the implementation of the VisCoIN model, a method that relies on a 
generative model (StyleGAN2) and a classifier (ResNet50), developed 
for visualizing the concepts activated during image classification.
The first objective is to assess the reproducibility of the published results on 
the CUB-200 dataset. The second objective is to evaluate the transferability of the
VisCoIN model to medical images, specifically using the NCT-CRC-HE dataset, and 
to assess its relevance as a tool for understanding the decision-making 
process of deep learning models in the context of medical image analysis. 
We were able to reproduce the published results to a certain extent. However, the
obtained FID score was significantly worse than the one reported in the original paper, leading 
to a struggling reconstruction, and we provide possible explanations for this discrepancy.
Transferring the VisCoIN model to medical images proved to be more difficult. While we
obtained some results indicating that the methodology is relevant, the overall architecture 
likely needs to be adaped to better suit the characteristics of medical images in order to produce more
usable outcomes.

Link to the original VisCoIN publication: https://arxiv.org/abs/2407.01331

Link to the original StyleGan publication: https://arxiv.org/abs/2006.06676 


## Git structure and sources

Link to the original StyleGan repository: https://github.com/NVlabs/stylegan2-ada-pytorch

This code was used with no modifications for the generator training.

Link to the original VisCoIN repository: https://github.com/GnRlLeclerc/VisCoIN
Code was adapted and modified for the classifier and VisCoIN training, to allow for new dataset usage and variation in experiments.

Scripts in the root /datasets folder were created for this project. 


````bash
├── ../stylegan2_ada  # Pytorch implementation of StyleGAN2 ADA
│
├── cli            # Command line functions
│
├── datasets       # Pytorch dataloaders for every dataset
│   ├── cub                 # CUB dataset loader
│   └── transforms          # Standard dataset preprocessing transformations
│
├── models         # Model definitions
│   ├── classifiers         # Classifier model (`f`)
│   ├── concept_extractors  # From classifier latent to concepts (`Psi` in VisCoIN)
│   ├── explainers          # From concepts to class (`Theta` in VisCoIN)
│   └── gan                 # StyleGAN implementation (modified stylegan2_ada)
│
├── testing        # Testing functions
│   ├── classifiers         # Testing function for the classifier
│   └── viscoin             # Testing functions for the viscoin ensemble
│
└── training       # Training functions
    ├── classifiers         # Training function for the classifier
    ├── losses              # Loss functions
    ├── utils               # Training utilities
    └── viscoin             # Training function for the viscoin ensemble
````
-> ajouter dossier datasets

## Set up

We recommand using two separate environements for the GAN training and for other tasks. 

CUDA version ?

viscoin env -> use method from repo






## Get dataset and preprocess images

#### - Download and resize images for CUB200 dataset

The dataset used is downloaded from : https://www.vision.caltech.edu/datasets/cub_200_2011/

```bash
cd datasets/CUB-200
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
tar -xvzf CUB_200_2011.tgz
python download_cub_200_dataset.py
```

#### - Download and resize images for CRC dataset 

The dataset used is downloaded from : https://huggingface.co/datasets/1aurent/NCT-CRC-HE

```bash
cd datasets/NCT-CRC-HE
python download_nct_crc_he_dataset.py
```
The path to store the data can be adjusted in download_nct_crc_he_dataset.py and in download_cub_200_dataset.py if needed. 

## GAN training

train generator
````bash
source env_gan bin/activate
````

Get the starting checkpoint :

starting GAN : 
````bash
cd /stylegan2_ada

python train.py --outdir=training-runs --data=path_to_datasets/CUB200/CUB_200_2011/resized_images_256_for_gan --gpus=1 --resume=path_to_VisCoIN/checkpoints/ffhq140k-paper256-ada.pkl --kimg=500 --snap=10
````


## Classifier training

Next steps

````bash
source env_viscoin bin/activate
````
train classifier



## VisCoIN training


train viscoin

visualize results (make a notebook)

