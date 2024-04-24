from dataset.datasets import CelebA
from model.unet import Unet
from diffusion.ddpm import DDPM
from pathlib import Path

# Local Libraries
from utils import utils
from dataset import datasets
from model.transformer import dit

from diffusion import ncsn
import matplotlib.pyplot as plt

import os
import torch
import torchvision
import numpy as np

# Load Configuration
config = utils.load_yaml("configuration/mnist.yaml")
config = utils.dict2namespace(config)

# Initialize Dataset & Dataloader
ds = datasets.MNIST(config)
dl = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, shuffle=True)


# Initialize U-Net
model = Unet(config, dim=config.model.dim)
model.to(config.training.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)

step = 0

sigmas = torch.tensor(
    np.exp(np.linspace(np.log(1), np.log(0.01),
                        10))).float().to(config.training.device)

for epoch in range(config.training.epochs):
    for step, X in enumerate(dl):
        X = X.to(config.training.device)
        X = X / 256. * 255. + torch.rand_like(X) / 256.
        
        labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
        loss = ncsn.anneal_dsm_score_estimation(model, X, labels, sigmas, 2)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss.item())
    grid_size = 5
    samples = torch.rand(grid_size ** 2, 1, 64, 64, device=config.training.device)
    images = ncsn.anneal_Langevin_dynamics(samples, model, sigmas, 100, 0.00002)
    print("Generated images")