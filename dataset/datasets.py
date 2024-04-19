import torch
import torchvision
import os
from PIL import Image


class MNIST(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        
        # Transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(config.dataset.img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
        ])
        
        # Initialize Dataset
        self.dataset = torchvision.datasets.MNIST(
            root='./dataset/mnist', 
            train=False, 
            download=config.dataset.download, 
            transform=None)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        img = self.dataset[index]
        return self.transform(img[0])
        

class CelebA(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size):
        super().__init__()
        
        # Create Filepath Array
        self.img_dir = img_dir
        self.img_files = os.listdir(self.img_dir)
        self.img_filepaths = [os.path.join(self.img_dir, filename) for filename in self.img_files if filename.lower().endswith('.jpg')]


        # Transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int):
        img = Image.open(self.img_filepaths[index])
        return self.transform(img)