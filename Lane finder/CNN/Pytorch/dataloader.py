import torch
import numpy as np
import torchvision

# For transformations on masks and images
from PIL import Image
from skimage.transform import resize

# import my config/settings
from config import config
CFG = config.cfg
new_size = CFG.new_size
batch_size = CFG.batch_size

def train_image_loader(img_files, mask_files):
    train_dataset = ImageDataset(img_files, mask_files)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory = True,
        shuffle=False
    )
    return train_loader

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, target_paths):
        self.image_paths = image_paths
        self.target_paths = target_paths

    def transform(self, image, mask):

        # Resize image
        resize_img = torchvision.transforms.Resize(new_size)
        image = resize_img(image)

        # Convert mask to binary image and extract only the channel we need
        mask = np.asarray(mask).astype(np.int32)
        image = np.asarray(image).astype(np.int32).transpose(2, 0, 1)
        mask = np.array(mask[:,:,0]) == 255

        # Resize                                           ### NEEDS FIX ===========
        mask = resize(255*mask, new_size)
        mask = (mask > np.min(mask)*100).reshape(1, *mask.shape)

        return image, mask

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        x, y = self.transform(image, mask)
        return x, y

    def __len__(self):
        return len(self.target_paths)



def val_image_loader(img_files):
    train_dataset = ValImageDataset(img_files)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory = True,
        shuffle=False
    )
    return train_loader

class ValImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def transform(self, image):

        # Resize image
        resize_img = torchvision.transforms.Resize(new_size)
        image = resize_img(image)
        image = np.asarray(image).astype(np.float32).transpose(2, 0, 1)

        return image

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)
