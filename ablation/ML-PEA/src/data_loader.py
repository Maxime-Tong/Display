import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from glob import glob

class ImageDataset(Dataset):

    def __init__(self, mode, dim=None, opt=None):

        self.batch_size = opt.batch_size
        self.mode = mode
        if mode == "train":
            self.dataset_path = os.path.join("./{}".format(opt.dataset), "train")
        else:
            self.dataset_path = os.path.join("./{}".format(opt.dataset), "test")

        self.img_pths = glob(os.path.join(self.dataset_path, "*"))
        self.dim = dim

    def __getitem__(self, index):

        self.image_path = self.img_pths[index]
        img = Image.open(self.image_path).convert("RGB")
        
        if self.mode == "train":
            transform_list = [
                transforms.ToTensor(),
                transforms.RandomCrop(self.dim),
            ]
        else:
            transform_list = [
                transforms.ToTensor(),
            ]
            
        transform_img = transforms.Compose(transform_list)
        img_tensor = transform_img(img).float()

        sample_dict = {
            "img": img_tensor,
            "path": self.image_path,
        }

        return sample_dict

    def __len__(self):
        return len(self.img_pths)
