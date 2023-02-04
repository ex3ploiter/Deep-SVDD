from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from .Attack import *
# from A

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os



class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0,attack_type='fgsm',attack_target='clear'):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader,attack_type,attack_target)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        
        self.attack_type=attack_type
        self.attack_target=attack_target
        

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(dist, self.nu), device=self.device)

                loss_epoch += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()


        results={}
        results['clear_AUC']=[]
        # results['Attack_Type'] =[]
        # results['Attack_Target'] =[]
        results['ADV_AUC'] =[]
        
        for time in range(0,5):
            # Set device for network
            net = net.to(self.device)
            
            
            
            batch_sz=64 if self.attack_type=='clear' else 1

            # Get test data loader
            _, test_loader = dataset.loaders(batch_size=batch_sz, num_workers=8)

            # Testing
            logger.info('Starting testing...')
            start_time = time.time()
            idx_label_score = []
            idx_label_score_adv = []
            net.eval()
            # with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist
                    
                
                shouldBeAttacked=False
                if self.attack_target=='normal':
                    if labels==0:
                        shouldBeAttacked=True
                elif self.attack_target=='anomal':
                    if labels==1:
                        shouldBeAttacked=True
                elif self.attack_target=='both':
                    shouldBeAttacked=True
                
                
                
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                
                ########---------------########
                
                if self.attack_type!='clear':
                    
                # if self.attack_target=='normal':
                    if shouldBeAttacked==True:
                        if self.attack_type=='fgsm':
                            adv_delta=fgsm(net,inputs,self.c,0.1,self.objective,self.R)
                        
                        if self.attack_type=='pgd':
                            adv_delta=pgd(net, inputs, self.c, 0.1, 1/255, 10,self.objective,self.R)
                        
                        adv_image = inputs+adv_delta if labels==0 else inputs-adv_delta

                        
                        adv_outputs=net(adv_image)
                        dist = torch.sum((adv_outputs - self.c) ** 2, dim=1)
                        if self.objective == 'soft-boundary':
                            scores = dist - self.R ** 2
                        else:
                            scores = dist
                        
                            
                idx_label_score_adv += list(zip(idx.cpu().data.numpy().tolist(),
                            labels.cpu().data.numpy().tolist(),
                            scores.cpu().data.numpy().tolist()))



            self.test_time = time.time() - start_time
            logger.info('Testing time: %.3f' % self.test_time)

            self.test_scores = idx_label_score

            # Compute AUC
            _, labels, scores = zip(*idx_label_score)
            labels = np.array(labels)
            scores = np.array(scores)
            
            

            self.test_auc = roc_auc_score(labels, scores)
            logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))
            
            results['clear_AUC'].append( 100. * self.test_auc)

            logger.info('Finished testing.')
            
            
    # ----

            _, labels, scores = zip(*idx_label_score_adv)
            labels = np.array(labels)
            
            scores = np.array(scores)
            
            

            self.test_auc = roc_auc_score(labels, scores)
            logger.info('ADV : Test set AUC : {:.2f}%'.format(100. * self.test_auc))

            logger.info('ADV : Finished testing.')        
            
            
            
            # results['Attack_Type'].append( self.attack_type)
            # results['Attack_Target'].append(  self.Attack_Target)
            
            
            results['ADV_AUC'].append( 100. * self.test_auc)
            
        df = pd.DataFrame(results)
        df.to_csv(os.path.join('./',f'Results_{self.attack_type}_{self.attack_target}'), index=False)
        

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, _ = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
