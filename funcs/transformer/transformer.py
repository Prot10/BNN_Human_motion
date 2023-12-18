from typing import Any, Callable, Optional, Union, Tuple
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from .sttformer import STTFormerBayes
from ..utils.loss_funcs import mpjpe_error
import numpy as np 
import scipy 
import torch
from bayesian_torch.models.dnn_to_bnn import get_kl_loss

input_n  =10 # number of frames to train on (default=10)
output_n =25 # number of frames to predict on
input_dim=3  # dimensions of the input coordinates(default=3)
skip_rate=1  # skip rate of frames
joints_to_consider=22
lr = 1e-02 # learning rate
use_scheduler=True # use MultiStepLR scheduler
milestones=list(range(0,35,3))  # the epochs after which the learning rate is adjusted by gamma
gamma=0.1 #gamma correction to the learning rate, after reaching the milestone epochs
weight_decay=1e-05 # weight decay (L2 penalty)
dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
num_joints = len(dim_used) // 3

class LitSTTFormerBayes(L.LightningModule):
    def __init__(self, repetitions = 100, speeds = True, **kwargs) -> None:
        super().__init__()
        self.transformer = STTFormerBayes(repetitions = repetitions, **kwargs)
        self.reps = repetitions
        self.speeds = speeds

    def forward(self, x, return_population=False, return_dev = False) -> Any:
        sequences_train = x[:, 0:input_n, dim_used].view(-1,input_n,num_joints,3)
        start_place = sequences_train[:,[input_n-1],:,:]
        X = sequences_train
        if self.speeds:
            X = sequences_train.diff(dim=1, prepend = sequences_train[:,[0],:,:])
        X = X.permute(0,3,1,2)
        
        y, kl = self.transformer.forward(X)
        y = y.view(self.reps, -1, output_n, num_joints, 3)

        if self.speeds:
            y = y.cumsum(dim=2) + start_place.unsqueeze(0)

        if return_population:
            return y, kl
        
        y_mu = y.mean(dim=0)
        if not return_dev:
            return y_mu, kl
        dev = torch.linalg.vector_norm(y - y_mu, dim=-1)
 
        return y_mu, dev, kl
        
    
    @torch.no_grad
    def build_ci(self, x: torch.Tensor, alpha:float=0.1, bonferroni:bool=True):
        if bonferroni:
            alpha = alpha/x.shape[-2]
            # correct for the number of joints
        y_samples, _ = self.forward(x, return_population=True)
        # sample is dimension 1000 x Batch x OutputFrames x Joints x 3
        mu = y_samples.mean(dim=0, keepdim=True)
        diff = (y_samples - mu)
        dev = torch.linalg.vector_norm(diff, dim=-1)
        dev = dev.numpy(force=True)
        return mu.squeeze(), np.quantile(dev, 1-alpha, axis=0)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss, kl_loss, avu_loss = self.step(batch)
        self.log('training_loss', loss)
        self.log('KL_divergence', kl_loss)
        self.log('AvU_loss', avu_loss)
        return loss + kl_loss / batch.shape[0] + avu_loss/4
    
    def step(self, batch : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.float()
        y_pred, y_dev , kl_loss = self.forward(batch, return_dev=True)
        y_pred = y_pred.view(-1,output_n,num_joints,3)

        y_gt = batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,num_joints,3)
        error = torch.linalg.vector_norm(y_gt - y_pred, dim=-1)
        mpjpe = error.mean()
        avu_loss = ((error - y_dev)**2).mean().sqrt()

        return mpjpe, kl_loss, avu_loss


    def configure_optimizers(self) -> OptimizerLRScheduler:
        avoid_regularization = ['mu', 'rho', 'bias', '.1.weight', '.1.bias']
        deterministic_params = [p for n,p in self.named_parameters() if all(p not in n for p in avoid_regularization)]
        stochastic_params = [p for n,p in self.named_parameters() if any(p in n for p in avoid_regularization)]
        optimizer_params = [
            {'params': deterministic_params, 'weight_decay':weight_decay},
            {'params': stochastic_params,    'weight_decay': 0 }
        ]
        optimizer = torch.optim.Adam(optimizer_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def validation_step(self, batch, batch_idx):
        loss, _, avu_loss = self.step(batch)
        self.log('validation_loss', loss)
        self.log('AvU_loss_validation', avu_loss)

    # @torch.no_grad
    # def mvn_fit(self, x : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     y_samples = torch.stack([self.forward(x) for _ in range(self.reps)])
    #     # sample is dimension 1000 x Batch x OutputFrames x Joints x 3
    #     mu = y_samples.mean(dim=0, keepdim=True)
    #     dev = y_samples - mu
    #     S = torch.einsum('rbofi, rbofj -> bofij', dev, dev) / (self.reps-1)
    #     return mu, S
    

