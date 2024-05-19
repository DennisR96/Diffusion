from datasets.CelebA import CelebA
from model.unet import Unet
from model.transformer.dit import DiT_B_8
from diffusion.ddpm import DDPM
from pathlib import Path

# Local Libraries
from utils import utils
from model.transformer import dit

import os
import torch
import torchvision

# Load Configuration
config = utils.load_yaml("configuration/celebA.yaml")
config = utils.dict2namespace(config)

# Initialize Dataset & Dataloader
ds = CelebA(config)
dl = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, shuffle=True)

# Initialize U-Net
model = Unet(config, dim=config.model.dim)
model.to(config.training.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)

# Diffusion Algorithm 
diffusion = DDPM(timesteps=config.diffusion.timesteps, device=config.training.device)

loss = []
for epoch in range(config.training.epochs):
    for step, batch in enumerate(dl):
      optimizer.zero_grad()

      batch_size = batch.shape[0]
      batch = batch.to(config.training.device)

      # Sample Timestep Tensor
      t = torch.randint(0, config.diffusion.timesteps, (batch_size,), device=config.training.device).long()

      loss = diffusion.p_losses(model, batch, t)
      print(loss.item())
     

      loss.backward()
      optimizer.step()
    print(loss.item())
    imgs = diffusion.sample(model, image_size=config.dataset.img_size, batch_size=batch_size, channels=config.dataset.channels)
    torchvision.utils.save_image(imgs[-1], fp=f"./results/{config.dataset.name}_{epoch}.png")
      
    
