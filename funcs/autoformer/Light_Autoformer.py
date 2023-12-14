from typing import Any, Callable, Optional, Union
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from .Autoformer import Model as autoformer
from ..utils.loss_funcs import mpjpe_error
import numpy as np 
import torch

input_n=10 # number of frames to train on (default=10)
output_n=25 # number of frames to predict on
input_dim=3  # dimensions of the input coordinates(default=3)
skip_rate=1  # skip rate of frames
joints_to_consider=22
lr = 1e-04 # learning rate
use_scheduler=True # use MultiStepLR scheduler
milestones=[10,20,30]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.1 #gamma correction to the learning rate, after reaching the milestone epochs
weight_decay=1e-07 # weight decay (L2 penalty)
dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

class LitAutoformer(L.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.autoformer = autoformer(configs)

    def forward(self, x) -> Any:
        return self.autoformer.forward(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss,kl = self.step(batch)
        self.log_dict({'training mpjpe':loss,'kl':kl},on_step=True,on_epoch=True,prog_bar=True)
        return loss+kl
    
    def step(self, batch) -> torch.Tensor:
        batch = batch.float()

        sequences_train=torch.cat((torch.zeros(*batch[:,:1,dim_used].size()).to(batch.device),batch[:,1:input_n,dim_used]-batch[:,:input_n-1,dim_used]),1)
        sequences_gt=batch[:,input_n:,dim_used]

        sequences_predict, kl = self.autoformer.forward(sequences_train)
        sequences_predict[:,1:,:]=sequences_predict[:,1:,:]+sequences_predict[:,:output_n-1,:]
        sequences_predict=sequences_predict+batch[:,input_n-1:input_n,dim_used]

        return mpjpe_error(sequences_predict,sequences_gt), kl


    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def validation_step(self, batch, batch_idx):
        loss,kl = self.step(batch)
        self.log_dict({'validation mpjpe':loss,'kl':kl},on_step=True,on_epoch=True,prog_bar=True)
        return loss
