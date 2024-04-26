from diffusion import ddpm
import torch

N = 32
device = "mps"
timesteps = 1000


# Create a torch.Tensor of [N] with Random Timesteps
t = torch.randint(0, timesteps, (N,), device=device).long()

# Load Diffusion Model
diffusion = ddpm.DDPM(t, "mps")

# Load an Image Batch


print()