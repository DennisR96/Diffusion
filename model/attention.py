from torch import nn, einsum
import torch
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

class Attention(nn.Module):
    """
    Module for the Attention Mechanism
    """
    def __init__(self, dim, heads=4, dim_head=32):
        '''
        Initializes the Attention module

        Args:
        dim (int): Dimensionality of the embeddings.
        heads (int) : Number of Attention heads
        dim_heads (int) : Dimensionality of each attention head.
        '''
        super().__init__()
        
        # Dot Product Scaling Factor
        self.scale = dim_head**-0.5
        
        # Number of Attention Heads             
        self.heads = heads
        
        # Hidden Layer Dimensionality
        hidden_dim = dim_head * heads
        
        # Linear Transform for Query, Key and Value
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        
        # Linear Transform for the Output
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        # Input Shape
        b, c, h, w = x.shape
        
        # Linear Transform and Split into Query, Key and Value – W^Q W^K W^V
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        
        # Query Scaling – 1/sqrt(d)
        q = q * self.scale

        # Calculate the Attention Scores – Query dot Key
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        
        # Softmax Stabilization
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()                 # Stabilize by Substracting max value
        attn = sim.softmax(dim=-1)                                          # Apply Softmax

        # Calculate Product of Attention Scores and Values
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        
        # Rearrange and Shape Output
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        
        # Linear Transformation Output
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        '''
        Initializes the Attention module

        Args:
        dim (int): Dimensionality of the embeddings.
        heads (int) : Number of Attention heads
        dim_heads (int) : Dimensionality of each attention head.
        '''
        super().__init__()
        
        # Dot Product Scaling Factor
        self.scale = dim_head**-0.5
        
        # Number of Attention Heads       
        self.heads = heads
        
        # Hidden Layer Dimensionality
        hidden_dim = dim_head * heads
        
        # Linear Transform for Query, Key and Value
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

         # Linear Transform for the Output        
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
         # Input Shape
        b, c, h, w = x.shape
        
        # Linear Transform and Split into Query, Key and Value – W^Q W^K W^V
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        # Softmax to Q and K
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        # Query Scaling – 1/sqrt(d)
        q = q * self.scale
        
        # Dot Product Key and Value
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # Dot Product Context and Q
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        
        # Rearrange Output Tensor
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        
        # Linear Transformation Out
        return self.to_out(out)
