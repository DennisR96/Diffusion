import torch
import torchvision
import os
from PIL import Image

class MNIST(torch.utils.data.Dataset):
    """
    Custom dataset class for MNIST that includes image resizing, normalization, and standardization.
    Designed to be compatible with PyTorch's Dataset interface.
    """

    def __init__(self, config):
        '''
        Initializes the dataset with specific transformations and loads MNIST data.

        Args:
        config: Configuration object containing parameters like image size and download flag.
        '''
        super().__init__()

        # Define transformations: resize, convert to tensor, normalize to range [-1, 1]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.dataset.img_size),                  # Resize images to specified size
            torchvision.transforms.ToTensor(),                                       # Convert image to tensor
            torchvision.transforms.Lambda(lambda t: (t * 2) - 1),                    # Normalize tensors to [-1, 1]
        ])
        
        # Load the MNIST dataset from local files if available, or download it otherwise
        self.dataset = torchvision.datasets.MNIST(
            root='./dataset/mnist',                                                   # Set the root directory for the dataset
            train=False,                                                              # Use the test split of the dataset
            download=config.dataset.download,                                         # Flag to download the dataset if not locally available
            transform=None)                                                           # Apply no initial transformations (transformations are applied on-the-fly in __getitem__)

    def __len__(self):
        '''
        Returns the total number of images in the dataset.

        Returns:
        int: The number of items in the dataset.
        '''
        return len(self.dataset)  # Return the size of the dataset
    
    def __getitem__(self, index: int):
        '''
        Retrieves an image by index, applies transformations, and returns the processed image.

        Args:
        index (int): The index of the data item to retrieve.

        Returns:
        Tensor: The transformed image tensor.
        '''
        img, _ = self.dataset[index]  # Retrieve image and label, ignore the label
        return self.transform(img)  # Apply transformations to the image and return it