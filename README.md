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

The detailed report of the experiments is available in ```VisCoIN_implementation_medical_images.pdf```

Link to the original VisCoIN publication: https://arxiv.org/abs/2407.01331

Link to the original StyleGan publication: https://arxiv.org/abs/2006.06676 


## Git structure and sources

Link to the original StyleGan repository: https://github.com/NVlabs/stylegan2-ada-pytorch

This code was used with no modifications for the generator training.

Link to the original VisCoIN repository: https://github.com/GnRlLeclerc/VisCoIN

This code was adapted and modified for the classifier and VisCoIN training, to allow for new dataset usage and variation in experiments.

Scripts in the root ```/datasets``` folder were created for this project. 

 <strong>The code that was written for this project is tagged as "PERSONAL WORK" below.</strong>

 Global structure of the repository:


```bash
datasets # PERSONAL WORK - scripts to download and preprocess CUB-200 and NCT-CRC-HE datasets
├── CUB-200
│   └── download_cub_200_dataset.py # PERSONAL WORK 
├── NCT-CRC-HE
│   └── download_nct_crc_he_dataset.py # PERSONAL WORK
VisCoIN # main folder for viscoin training
├── stylegan2_ada # code from the original NVIDIA repo, unmodified 
│
├── viscoin # scripts for viscoin training, adapted from the original VisCoIN repo
│        │
│        ├── datasets       # Pytorch dataloaders
│        │   ├── cub                 # CUB dataset loader
│        │   └── transforms          # Standard dataset preprocessing transformations
│        │   └── custom_local_dataset # PERSONAL WORK - loader for local NCT-CRC_HE dataset
│        │
│        ├── models         # Model definitions
│        │   ├── classifiers         # Classifier model (`f`)
│        │   ├── concept_extractors  # From classifier latent to concepts (`Psi` in VisCoIN)
│        │   ├── explainers          # From concepts to class (`Theta` in VisCoIN)
│        │   └── gan                 # StyleGAN implementation (modified stylegan2_ada)
│        │
│        ├── testing        # Testing functions
│        │   ├── classifiers         # Testing function for the classifier
│        │   └── viscoin             # Testing functions for the viscoin ensemble
│        │
│        ├── training       # Training functions
│        │   ├── classifiers         # Training function for the classifier
│        │   ├── losses              # Loss functions
│        │   ├── utils               # Training utilities
│        │   ├── viscoin      # PERSONAL WORK - adaptation of the training function for the viscoin
│        │   └── viscoin_custom   # PERSONAL WORK - adaptation for custom parameters
│        │
│        └── utils       # Utility functions
│            ├── gradcam         
│            ├── images              
│            ├── logging              
│            ├── maths               
│            ├── metrics               
│            └── types        
│
├── train_classifier_CUB.py  # PERSONAL WORK - script to train the classifier on the CUB-200 dataset
├── train_classifier_NCT_CRC_HE.py # PERSONAL WORK - script to train the classifier on the NCT_CRC_HE dataset
├── train_viscoin_CUB.py  # PERSONAL WORK - script to train the VisCoIN model on the CUB-200 dataset
├── train_viscoin_NCT_CRC_HE.py  # PERSONAL WORK - script to train the VisCoIN model on the NCT_CRC_HE datase
├── analysis.ipynb  # PERSONAL WORK - notebook to analyse the results after training
└── VisCoIN_implementation_medical_images.pdf  # PERSONAL WORK - complete report of the project
````


## Set up

We recommend using two separate environments for the GAN training and for other tasks (classsifier training and VisCoIN training).

* GAN training environment

We report here dependencies that worked for us, adapted from recommendations in the original repository (https://github.com/NVlabs/stylegan2-ada-pytorch):

````bash
conda create -n env_gan python=3.7
conda activate env_gan
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 psutil scipy 
````

* VisCoIN training environment

We report here the method from the original repository (https://github.com/GnRlLeclerc/VisCoIN):
```bash
conda env create -f conda.yml  # Create the `viscoin` environment
conda activate viscoin         # Activate it
```


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

Activate the GAN environment:
````bash
conda activate env_gan
````

Get the starting checkpoint that was used in this project:
````bash
cd VisCoIN
mkdir checkpoints
cd ./checkpoints
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/paper-fig7c-training-set-sweeps/ffhq140k-paper256-ada.pkl
````

Train the GAN with 1M images on the CUB-200 dataset:
````bash
cd /stylegan2_ada

python train.py --outdir=training-runs --data=path_to_datasets/CUB200/resized_images_256 --gpus=1 --resume=path_to_VisCoIN/checkpoints/ffhq140k-paper256-ada.pkl --kimg=1000 --snap=10
````


Train the GAN with 1M images on the NCT-CRC-HE dataset:
````bash
cd /stylegan2_ada

python train.py --outdir=training-runs --data=path_to_datasets/NCT-CRC-HE/resized_images_256 --gpus=1 --resume=path_to_VisCoIN/checkpoints/ffhq140k-paper256-ada.pkl --kimg=1000 --snap=10
````


## Classifier training


````bash
conda activate viscoin
cd VisCoIN
python train_classifier_NCT_CRC_HE.py # or train_classifier_CUB.py
````

## VisCoIN training

You will need to use a Cuda version of 10 or 11 (incompatibility with CUDA 12)

````bash
cd VisCoIN
python train_viscoin_NCT_CRC_HE.py # or train_viscoin_CUB.py
````

## Visualize results 

A notebook is provided to display results after training of the classifier, the GAN and the VisCoIN model on the NCT-CRC-HE dataset : ```analysis.ipynb```.

