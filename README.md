# Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task 

## Introduction
This project is the codebase for the paper titled "Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task," presented at NeurIPS 2023. The code implements Denoising Diffusion Probabilistic Models (DDPM) and is built using PyTorch.

## Paper
- **Title:** Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task 
- **Authors:** Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka 
- **Venue:** Advances in Neural Information Processing Systems (NeurIPS)
- **Published:** 2023 
- **Link:** [arXiv link](https://arxiv.org/abs/2310.09336)

## Installation
- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm
- einops

## Dataset 

You can download the datasets required for this project: 

- **Synthetic Data:**  
  The synthetic data for this project can be downloaded from the following link:  
  [Download Synthetic Data](https://www.dropbox.com/scl/fi/6zzb5h4bly2gbignwn4yz/single-body_2d_3classes.zip?rlkey=0uizen48trsl6cm4oaui2ze41&dl=0)

- **Real Data:**  
  The preprocessed CelebA data for this project can be downloaded from the following link:  
  [Download Preprocessed Data](https://www.dropbox.com/scl/fi/kikre4mhv3iipzxuytbrb/celeba-3classes-10000.zip?rlkey=4gllwowbxs6vers9abcoraz5g&dl=0)


## Usage
First, create an `input` directory at the root level of this project. Then, place the data files under the `input` directory.

To train the model with the synthetic data, run the `train.py` script with the desired parameters. For example:

`python3 train.py --dataset single-body_2d_3classes`

To train the model using the CelebA dataset:

`python3 train.py --dataset celeba-3classes-10000`


## Structure
- `train.py`: Main script for training the DDPM model.
- `load_dataset.py`: Script for loading and processing datasets.


## References
- The `DDPM` class in `train.py` is based on the implementation found at [TeaPearce/Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py).
- The `CrossAttention` class in `train.py` is inspired by the code in [Animadversio/DiffusionFromScratch/StableDiff_UNet_model.py](https://github.com/Animadversio/DiffusionFromScratch/blob/master/StableDiff_UNet_model.py).


