import math
import torch
import numpy as np

class SinusoidalPositionEmbeddings(torch.nn.Module):
    """
    Module to generate sinusoidal position embeddings for temporal data.
    """
    def __init__(self, dim):
        '''
        Initializes the SinusoidalPositionEmbeddings module.

        Args:
        dim (int): Dimensionality of the embeddings.
        '''
        super().__init__()
        self.dim = dim

    def forward(self, time):
        '''
        Generates sinusoidal position embeddings for the input time tensor.

        Args:
        time (Tensor): Input tensor representing time with shape [batch_size, 1].

        Returns:
        Tensor: Sinusoidal position embeddings with shape [batch_size, dim].
        '''
        # Extract the Device
        device = time.device                

        # Calculate half the dimension for cosine and sine embeddings
        half_dim = self.dim // 2
        
        # Compute the exponential factor for sinusoidal embeddings
        embeddings = math.log(10000) / (half_dim - 1)
        
        # Generate exponential values for embeddings
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # Compute sinusoidal embeddings
        embeddings = time[:, None] * embeddings[None, :]
        
        # Concatenate sine and cosine embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        return embeddings

# Testing Example
## Generate Time Tensor from 0 to 1 with a Sequence of N=5 
time = torch.tensor(np.linspace(0, 1, 24))
    
Test = SinusoidalPositionEmbeddings(dim=4)
embeddings = Test(time)
print(embeddings)