# Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task 

## Introduction
This project is the code for the paper "Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task" (NeurIPS 2023). 
This code is Denoising Diffusion Probabilistic Models (DDPM) and is built using PyTorch. 

## Paper
**Title:** Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task 
**Authors:** Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka 
**Venue:** NeurIPS 
**Published:** 2023 
**Link:** [arXiv link](https://arxiv.org/abs/2310.09336)

## Installation
- PyTorch
- torchvision
- numpy
- matplotlib
- PIL
- tqdm
- einops

## Dataset 

You can download the datasets required for this project: 

- **Synthetic Data:**  
  The synthetic data for this project can be downloaded from the following link:  
  [Download Synthetic Data](#add-your-link-here)

- **Preprocessed Data:**  
  The preprocessed CelebA data for this project can be downloaded from the following link:  
  [Download Preprocessed Data](#add-your-link-here)


## Usage
First, create an "input/" directory at the root level of this project. Then, place the data files under the "input" directory.

To train the model with the synthetic data, run the `train.py` script with the desired parameters. For example:

`python train.py --dataset single-body_2d_3classes`

To train the model using the CelebA dataset:

`python train.py --dataset celeba-3classes-10000`

## Structure
- `train.py`: Main script for training the DDPM model.
- `load_dataset.py`: Script for loading and processing datasets.


