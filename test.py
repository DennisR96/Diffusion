from dataset.datasets import CelebA
from model.unet import Unet
from diffusion.ddpm import DDPM
from pathlib import Path

# Local Libraries
from utils import utils
from dataset import datasets

import os
import torch
import torchvision

# Load Configuration
config = utils.load_yaml("configuration/mnist.yaml")
config = utils.dict2namespace(config)

# Initialize Dataset & Dataloader
ds = datasets.MNIST(config)
dl = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, shuffle=True)

for epoch in range(1): 
    for step, batch in enumerate(dl):
        batch