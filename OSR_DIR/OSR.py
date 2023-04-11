import torch
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import ImageFolder
import torch
from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.models import resnet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset 
import numpy as np
import time
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import glob
import os
import shutil
from glob import glob
import os
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd 
import json

from utils import MyDataset_Binary,auc_softmax,auc_softmax_adversarial,getLoaders
from Attack import TestTimePGD
import torch.optim as optim
from Models import Model
from tqdm import tqdm



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

from torchattacks import PGD
TrainTimePGD = PGD #(fln)



    
    





def TrainAndTest(idx,model,train_loader,test_loader,attack_eps,attack_steps,attack_alpha,num_epoches):

    attack_train = TrainTimePGD(model, eps=attack_eps,alpha=attack_alpha, steps=attack_steps)
    attack_test = TestTimePGD(model, eps=attack_eps,alpha=attack_alpha, steps=attack_steps, num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
    
    clean_auc=[]
    adv_auc=[]
    
    for epoch in range(num_epoches):
            total_loss, total_num = 0.0, 0
            loss = nn.CrossEntropyLoss()
            train_bar =  tqdm(train_loader, desc='Train  Binary Classifier ...')
            for (img1, Y ) in train_bar:
                model.train()
                #attack = PGD(model, eps=attack_eps,alpha= attack_alpha,steps=attack_steps, num_classes=6)
                adv_data = attack_train(img1, Y)
                optimizer.zero_grad()            
                out_1 = model(adv_data.cuda()) 
                loss_ = loss(out_1,Y.cuda())  
                loss_.backward()
                optimizer.step()
                total_num += adv_data.size(0)
                total_loss += loss_.item() * adv_data.size(0)
                total_num += train_loader.batch_size
                total_loss += loss_.item() * train_loader.batch_size
                train_bar.set_description(' Epoch :  {} ,   Loss: {:.4f}'.format(epoch ,  total_loss / total_num))
            #attack = PGD(model, eps=attack_eps,alpha= attack_alpha,steps=attack_steps, num_classes=6)
            auc=auc_softmax(model, test_loader, num_classes=6 )
            auc_adv=auc_softmax_adversarial(model, test_loader , attack_test, num_classes=6)    
            
            clean_auc.append(auc)
            adv_auc.append(auc_adv)
    
    df=pd.DataFrame({'Clean AUC' : clean_auc , 'Adv AUC' :adv_auc  })
    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    if not os.path.exists('./logs'):
        os.makedirs('./logs')


    
    df.to_csv(os.path.join(f'./Results',f'Results_Fashion_{idx}'))
    
    info_to_save = {
        'attack_eps': attack_eps,
        'attack_steps': attack_steps,
        'attack_alpha': attack_alpha,
        'num_epoches': num_epoches,
        'Loss': total_loss / total_num   
    }


    with open(os.path.join(f'./logs',f'logs_Fashion_{idx}'),'w') as f:
        json.dump(info_to_save, f)
            


labels_to_test=[(0, 1, 2, 3, 4, 5),
 (1, 3, 9, 2, 7, 0),
 (2, 6, 8, 4, 3, 0),
 (3, 9, 6, 7, 0, 1),
 (5, 2, 6, 9, 7, 0),
 (6, 5, 4, 1, 3, 0),
 (7, 9, 3, 4, 0, 1),
 (9, 2, 3, 6, 7, 0)]


model = Model('resnet18').cuda() #'resnet18'   'resnet50' 'wide_resnet50_2'

attack_eps = 8/255
attack_steps = 10
attack_alpha=0.00784313725490196

for idx,labelsToKeep in enumerate(labels_to_test):
    
    train_loader,test_loader=getLoaders(labelsToKeep)
    TrainAndTest(idx=idx,model=model,train_loader=train_loader,test_loader=test_loader,attack_eps=attack_eps,attack_alpha=attack_steps,attack_steps=attack_alpha,num_epoches=30)