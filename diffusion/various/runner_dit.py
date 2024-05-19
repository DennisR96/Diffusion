## Imports
# 
from datasets.CelebA import CelebA
from datasets.mnist import MNIST
from model.unet import Unet
from model.transformer.dit import DiT_B_8
from model.ema import EMA
from diffusion.ddpm import DDPM
from pathlib import Path

# Local Libraries
from utils import utils
from model.transformer import dit

import os
import torch
import torchvision

# Load Configuration
config = utils.load_yaml("configuration/CelebA.yaml")
config = utils.dict2namespace(config)

# Initialize Dataset & Dataloader
ds = CelebA(config)
dl = torch.utils.data.DataLoader(ds, batch_size=config.training.batch_size, shuffle=True)


# Initialize U-Net
model = Unet(config, dim=config.model.dim)
model.to(config.training.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.lr)

# Initalize DiT
model = DiT_B_8(input_size=config.dataset.img_size, in_channels=config.dataset.channels, learn_sigma=False).to("mps")
model.to(config.training.device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)

# Diffusion Algorithm 
diffusion = DDPM(timesteps=config.diffusion.timesteps, device=config.training.device)

loss = []

for epoch in range(config.training.epochs):
    model.train()
    for step, batch in enumerate(dl):
      optimizer.zero_grad()

      batch_size = batch.shape[0]
      batch = batch.to(config.training.device)

      # Sample Timesteps
      t = torch.randint(0, 2, (batch_size,), device=config.training.device).long()

      loss = diffusion.p_losses(model, batch, t)
      
      loss.backward()
      optimizer.step()
    print(f"Current Loss at Epoch {epoch} is: {loss.item()}")
    
    # Evaluate after Epoch
    
    model.eval()
    
    imgs = diffusion.sample(model, image_size=config.dataset.img_size, batch_size=batch_size, channels=config.dataset.channels)
    torchvision.utils.save_image(imgs[-1], fp=f"./results/{config.dataset.name}_{epoch}.png")
    
      
    
