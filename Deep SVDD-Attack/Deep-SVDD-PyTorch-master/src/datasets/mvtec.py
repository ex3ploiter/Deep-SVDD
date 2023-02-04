from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Dataset
import os
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
from glob import glob 

class MVTec_Dataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True):
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          


          self.image_files = normal_image_files+anomaly_image_files

        self.train = train
        
        

        

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1
        
        return image, target,index
        

    def __len__(self):
        return len(self.image_files)
    


