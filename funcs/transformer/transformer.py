from typing import Any, Callable, Optional, Union
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from .sttformer import STTFormerBayes
from ..utils.loss_funcs import mpjpe_error
import numpy as np 
import torch
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

input_n  =10 # number of frames to train on (default=10)
output_n =25 # number of frames to predict on
input_dim=3  # dimensions of the input coordinates(default=3)
skip_rate=1  # skip rate of frames
joints_to_consider=22
lr = 1e-01 # learning rate
use_scheduler=True # use MultiStepLR scheduler
milestones=[5,10,15,20,25,30]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.1 #gamma correction to the learning rate, after reaching the milestone epochs
weight_decay=1e-05 # weight decay (L2 penalty)
dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

class LitSTTFormerBayes(L.LightningModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.transformer = STTFormerBayes(**kwargs)

    def forward(self, x) -> Any:
        return self.transformer.forward(x)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.step(batch)
        kl_loss = get_kl_loss(self.transformer)
        self.log('training_loss', loss)
        self.log('KL_divergence', kl_loss)
        return loss + kl_loss / batch.shape[0]
    
    def step(self, batch) -> torch.Tensor:
        batch = batch.float()

        sequences_train = batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
        sequences_gt = batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

        sequences_predict = self.transformer.forward(sequences_train).view(-1, output_n, joints_to_consider, 3)

        return mpjpe_error(sequences_predict,sequences_gt)


    def configure_optimizers(self) -> OptimizerLRScheduler:
        deterministic_params = [p for p in self.parameters() if not hasattr(p, 'kl_loss')]
        stochastic_params = [p for p in self.parameters() if hasattr(p, 'kl_loss')]
        optimizer_params = [
            {'params': deterministic_params, 'weight_decay':weight_decay},
            {'params': stochastic_params, 'weight_decay':0}
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('validation_loss', loss)


