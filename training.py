from dataset.datasets import CelebA
from model.unet import Unet
from diffusion.ddpm import DDPM

import os
import torch

# Configuration
device = "mps"
epochs = 1
timesteps = 256

# Initialize Dataset
batch_size = 32
dataset = CelebA(img_size=(64, 64), img_dir="dataset/img_align_celeba/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize U-Net
model = Unet(dim=64)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Diffusion Algorithm 
diffusion = DDPM(timesteps=timesteps, device=device)

loss = []

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      batch_size = batch.shape[0]
      batch = batch.to(device)

      # Algorithm 1 line 3: sample t uniformally for every example in the batch
      t = torch.randint(0, timesteps, (batch_size,), device=device).long()

      loss = diffusion.p_losses(model, batch, t, loss_type="l1")
      print(loss.item())

      loss.backward()
      optimizer.step()