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

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        seq_len = x.size()[0]
        print(seq_len)

        return self.encoding[:seq_len, :]

time = torch.tensor(np.arange(0, 10000,1), dtype=int)

Test_B = SinusoidalPositionEmbeddings(dim=4)
Test = PositionalEncoding(d_model=4, max_len=10000, device="cpu")
embeddings = Test(time)
embeddings_2 = Test_B(time)
print(embeddings[50])
print(embeddings_2[50])