import torch
import torchvision
import os
from PIL import Image

class CelebA(torch.utils.data.Dataset):
    def __init__(self, config):
        super().__init__()
        
        # Create Filepath Array
        self.img_dir = config.dataset.img_dir
        self.img_files = os.listdir(self.img_dir)
        self.img_filepaths = [os.path.join(self.img_dir, filename) for filename in self.img_files if filename.lower().endswith('.jpg')]


        # Transformation
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config.dataset.img_size, config.dataset.img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: (t * 2) - 1),
        ])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int):
        img = Image.open(self.img_filepaths[index])
        return self.transform(img)