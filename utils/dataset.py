import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class imgDataset(Dataset):
    def __init__(self, root_dir, file_format='.png'):
        self.root_dir = root_dir
        # self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if file_format in f]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name)
        # if self.transform:
        #     image = self.transform(image)
        width, height = image.size
        pad_w = 32 - (width % 32)
        pad_h = 32 - (height % 32)
        image = TF.pad(image, (pad_w, pad_h, 0, 0))
        return TF.to_tensor(image)