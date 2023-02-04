from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms

import glob 
import os

class MVTec_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])

        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        self.train_set = MyMVTec(root=self.root, train=True,normal_class=normal_class,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        # train_idx_normal = get_target_label_idx(train_set.targets, self.normal_classes)
        # self.train_set = Subset(train_set, train_idx_normal)

        self.test_set = MyMVTec(root=self.root, train=False,normal_class=normal_class,
                              transform=transform, target_transform=target_transform)






class MyMVTec(TorchvisionDataset):
    def __init__(self, root, normal_class, transform=None, target_transform=None, train=True):
        self.transform = transform
        root=os.path.join(root,'mvtec_anomaly_detection')
        
        mvtec_labels=['bottle' , 'cable' , 'capsule' , 'carpet' ,'grid' , 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood','zipper']
        category=mvtec_labels[normal_class]

        if train:
            self.image_files = glob.glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob.glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob.glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          
        #   if normal:
        #     self.image_files = normal_image_files
        #   else:
        #     self.image_files = anomaly_image_files

          self.image_files = normal_image_files+anomaly_image_files

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
        
        return image, target, index
        

    def __len__(self):
        return len(self.image_files)
    
    