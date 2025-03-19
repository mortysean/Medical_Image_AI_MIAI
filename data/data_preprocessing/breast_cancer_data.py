import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class BreastCancerDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(img_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        
        # Define default transformation if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load images and masks as grayscale
        image = Image.open(img_path).convert("L")  
        mask = Image.open(mask_path).convert("L")

        # Apply transformations
        image = self.transform(image)  # Ensure image is tensor
        mask = transforms.ToTensor()(mask)  # Convert mask to tensor
        mask = (mask > 0.5).float()  # Binary mask (0 or 1)

        # Convert mask to a classification label (0 or 1)
        label = int(mask.sum() > 0)  # If any nonzero pixel exists, classify as 1 (cancer), else 0 (normal)

        return image, torch.tensor(label, dtype=torch.long)  # Ensure label is LongTensor
