from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Dataset
import os
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import glob 

class MVTec_Dataset(TorchvisionDataset):
    def __init__(self, root, normal_class, transform=None, target_transform=None, train=True):
        self.transform = transform
        
        mvtec_labels=['bottle' , 'cable' , 'capsule' , 'carpet' ,'grid' , 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood','zipper']

        category=mvtec_labels[normal_class]
        
        
        self.train_set = glob.glob(
            os.path.join(root, category, "train", "good", "*.png")
        )
      
        image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
        normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
        anomaly_image_files = list(set(image_files) - set(normal_image_files))

        self.test_set = normal_image_files+anomaly_image_files

        # self.train = train
        
        

        

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
    


