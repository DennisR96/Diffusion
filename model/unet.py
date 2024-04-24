import math
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import nn, einsum
import torch.nn.functional as F
from .embeddings import SinusoidalPositionEmbeddings
from .attention import Attention, LinearAttention 
from .helpers import exists, default, num_to_groups

class Residual(nn.Module):
    '''
    Residual Block for neural networks, particularly useful in deep networks to avoid
    the vanishing gradient problem by adding the input x directly to the output of
    a functional transformation of x.
    https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/ResBlock.png/1200px-ResBlock.png 
    '''

    def __init__(self, fn):
        '''
        Initializes the Residual block with a given function or neural network module (fn).

        Args:
        fn (callable): The operation to apply to the input, e.g., a combination of convolution,
                       normalization, and activation functions.
        '''
        super().__init__()  # Initialize the base nn.Module class
        self.fn = fn        # Assign the function to a class variable

    def forward(self, x, *args, **kwargs):
        '''
        Processes the input through the function 'fn' and adds the input x to the output of fn.

        Args:
        x (Tensor): The input tensor.
        *args, **kwargs: Additional parameters for the function 'fn'.

        Returns:
        Tensor: The output tensor after adding x to the transformed x by 'fn'.
        '''
        return self.fn(x, *args, **kwargs) + x 


def Upsample(dim, dim_out=None):
    '''
    Creates an upsampling module that doubles the resolution of input feature maps using nearest neighbor 
    interpolation followed by a convolutional layer to potentially adjust the number of output channels.

    Args:
    dim (int): The number of input channels.
    dim_out (int, optional): The number of output channels. Defaults to the same as input channels if not provided.

    Returns:
    nn.Sequential: A sequential container of an Upsample layer and a Conv2d layer.
    '''
    return nn.Sequential(
        # Upsample the input by a factor of 2 using nearest neighbor interpolation
        nn.Upsample(scale_factor=2, mode="nearest"),  
        
        # Apply a 2D convolution with specified in/out channels and a kernel size of 3x3 with padding set to 1.
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),  
    )


def Downsample(dim, dim_out=None):
    '''
    Creates a downsampling module that reduces the spatial dimensions of input feature maps by a factor of 2
    using a combination of rearrangement and convolution. This module effectively reduces the height and width
    while increasing the number of channels, optionally adjusting the number of output channels.

    Args:
    dim (int): The number of input channels.
    dim_out (int, optional): The number of output channels. If not provided, it defaults to the same as input channels.

    Returns:
    nn.Sequential: A sequential container of a Rearrange layer for downsampling and a Conv2d layer for channel adjustment.
    '''
    return nn.Sequential(
        # Rearrange input tensor to downsample spatial dimensions and increase channel dimensions,
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), 
        
        # Convolution Layer to adjust the channel count to the desired output dimension.
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)  # Convolutional layer to optionally adjust channel dimensions
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    A convolutional layer that implements Weight Standardization to improve training stability with Group Normalization.
    Weight Standardization standardizes the weights of the convolutional filters to have zero mean and unit variance,
    as suggested by the paper at https://arxiv.org/abs/1903.10520. This technique is beneficial when used alongside
    Group Normalization.
    """

    def forward(self, x):
        '''
        Overrides the forward pass to apply weight standardization to the weights before performing the convolution.

        Args:
        x (Tensor): Input tensor to the convolutional layer.

        Returns:
        Tensor: The output tensor after applying the convolution with standardized weights.
        '''
        # Set a small epsilon for numerical stability depending on the dtype of the input
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        # Calculate mean and variance of the weights
        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))

        # Standardize weights
        normalized_weight = (weight - mean) / (var + eps).rsqrt()

        # Perform convolution with standardized weights
        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

class Block(nn.Module):
    """
    A neural network block that combines a weight-standardized convolution, group normalization,
    and SiLU activation. Optionally applies a learnable scaling and shifting to the normalized output.
    This combination is commonly used in modern neural architectures for effective feature extraction and training stability.
    """

    def __init__(self, dim, dim_out, groups=8):
        '''
        Initializes the block with convolution, normalization, and activation layers.

        Args:
        dim (int): Number of input channels.
        dim_out (int): Number of output channels.
        groups (int, optional): Number of groups for group normalization. Defaults to 8.
        '''
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)  # Convolution layer with weight standardization
        self.norm = nn.GroupNorm(groups, dim_out)  # Group normalization layer
        self.act = nn.SiLU()  # SiLU activation function, also known as Swish

    def forward(self, x, scale_shift=None):
        '''
        Processes the input through the block, applying convolution, normalization, and optionally,
        a scale and shift transformation before the activation.

        Args:
        x (Tensor): The input tensor.
        scale_shift (tuple of Tensors, optional): A tuple containing scaling and shifting tensors. If provided,
                                                  these are used to modify the normalized output before activation.

        Returns:
        Tensor: The output tensor after processing through the block.
        '''
        x = self.proj(x)  # Apply weight-standardized convolution
        x = self.norm(x)  # Apply group normalization

        # If a scale and shift are provided, apply them
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift  # Apply learned scaling and shifting

        x = self.act(x)  # Apply SiLU activation
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class PreNorm(nn.Module):
    '''
    Applies layer normalization (specifically, Group Normalization with a group size of 1) 
    before passing the input tensor through a given function
    '''
    def __init__(self, dim, fn):
        '''
        
        Args:
        dim (int)       
        fn
        '''
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        config,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        self.channels = config.dataset.channels         # Image Channels
        self.self_condition = self_condition            # Condition
        
        
        # Double Input Channels if Conditional
        input_channels = self.channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, self.channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
