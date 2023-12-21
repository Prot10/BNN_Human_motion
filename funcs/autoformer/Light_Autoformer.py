from typing import Any, Callable, Optional, Union
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from .Autoformer import Model as autoformer
from .Autoformer_Enc_only import Model as autoformer_only, Small_decoder_bayes as bayes_dec, Small_decoder_freq as freq_dec
from ..utils.loss_funcs import mpjpe_error
import numpy as np 
import torch

input_n=10 # number of frames to train on (default=10)
output_n=25 # number of frames to predict on
input_dim=3  # dimensions of the input coordinates(default=3)
skip_rate=1  # skip rate of frames
joints_to_consider=22
use_scheduler=True # use MultiStepLR scheduler
dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

class LitAutoformer(L.LightningModule):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs=configs
        list_auto = {"autoformer":autoformer,"autoformer_only":autoformer_only}
        model=list_auto[configs.model]
        self.autoformer = model(configs)

        if configs.model=="autoformer_only":
            list_dec = {"bayes":bayes_dec,"freq":freq_dec}
            dec = list_dec[configs.dec]
            self.dec = dec(configs)

    def forward(self, x, reps=None) -> Any:
        if self.configs.dec=="freq":
          seq = self.dec.forward(*(self.autoformer.forward(x)))
          return seq                
        else:
          if reps==None:
              print(reps)
              seq, kl = self.dec.forward(*(self.autoformer.forward(x)))
              return seq, kl
          else:
              enc_s, enc_t = self.autoformer.forward(x)
              kl = 0
              seq = torch.empty((reps,x.shape[0],self.configs.pred_len,self.configs.c_out)).to(x.device)
              for i in range(reps):
                seq[i,:,:,:], kl_run = self.dec.forward(enc_s, enc_t)
                kl += kl_run
              return seq, kl/reps
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
                                   # primo frame <- tutto a zero
        sequences_train=torch.cat((torch.zeros((batch.shape[0],1,self.configs.c_out)).to(batch.device),
                                   # sottrai successivi ai precedenti e trovi lo spostamento step-by-step
                                   batch[:,1:input_n,dim_used]-batch[:,:input_n-1,dim_used]),1)
        sequences_gt=batch[:,input_n:,dim_used]

        if self.configs.dec=="bayes":
            # passagli le velocità
            seq_run, kl_out = self.forward(sequences_train,self.configs.samples)
            seq = seq_run.mean(0)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:output_n-1,:]
                                # somma l'ultimo elemento dei train frames                            
            mpjpe = mpjpe_error(seq + batch[:,input_n-1:input_n,dim_used],sequences_gt)
            self.log_dict({'training mpjpe':mpjpe,'kl':kl_out},on_step=True,on_epoch=True,prog_bar=True)
            return mpjpe+kl_out/self.configs.num_batches
        else:
            seq = self.forward(sequences_train)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:output_n-1,:]

                                # somma l'ultimo elemento dei train frames                            
            mpjpe = mpjpe_error(seq + batch[:,input_n-1:input_n,dim_used],sequences_gt)
            self.log_dict({'training mpjpe':mpjpe},on_step=True,on_epoch=True,prog_bar=True)
            return mpjpe

    def configure_optimizers(self) -> OptimizerLRScheduler:
        avoid_regularization = ['mu', 'rho', 'bias', '.1.weight', '.1.bias']
        deterministic_params = [p for n,p in self.named_parameters() if all(p not in n for p in avoid_regularization)]
        stochastic_params = [p for n,p in self.named_parameters() if any(p in n for p in avoid_regularization)]
        optimizer_params = [
            {'params': deterministic_params, 'weight_decay':self.configs.weight_decay},
            {'params': stochastic_params,    'weight_decay': 0 }
        ]
        
        optimizer = torch.optim.Adam(optimizer_params, lr=self.configs.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.configs.milestones, gamma=self.configs.gamma)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'validation mpjpe'}
    
    @torch.no_grad
    def build_ci(self, x:torch.Tensor, reps=200, alpha:float=0.1, bonferroni:bool=True):
        if bonferroni:
            alpha = alpha/x.shape[-2]
        sequences_train=torch.cat((torch.zeros((x.shape[0],1,x.shape[2])).to(x.device),x[:,1:self.configs.seq_len,:]-x[:,:self.configs.seq_len-1,:]),1)
        y_samples, _ = self.forward(sequences_train, reps)
        y_samples = y_samples
        y_samples[:,:,1:,:]=y_samples[:,:,1:,:]+y_samples[:,:,:self.configs.pred_len-1,:]+x[:,self.configs.seq_len-1:self.configs.seq_len,:]
        mu = y_samples.mean(dim=0, keepdim=True)
        diff = (y_samples - mu).view(reps,-1,self.configs.pred_len,len(dim_used)//3,3)
        dev = torch.linalg.vector_norm(diff, dim=-1)
        dev = dev.numpy(force=True)
        return mu.squeeze(), np.quantile(dev, 1-alpha, axis=0)
    
    @torch.no_grad
    def speed_step(self,batch):
        sequences_train=torch.cat((torch.zeros((batch.shape[0],1,self.configs.c_out)).to(batch.device),
                                   batch[:,1:self.configs.seq_len,dim_used]-batch[:,:self.configs.seq_len-1,dim_used]),1)
        if self.configs.dec=="bayes":
            seq_run, _ = self.forward(sequences_train,self.configs.samples)
            seq = seq_run.mean(0)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:self.configs.pred_len-1,:]+batch[:,self.configs.seq_len-1:self.configs.seq_len,dim_used]
            return seq
        else:
            seq = self.forward(sequences_train,self.configs.samples)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:self.configs.pred_len-1,:]+batch[:,self.configs.seq_len-1:self.configs.seq_len,dim_used]
            return seq
        
    @torch.no_grad
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
                                   # primo frame <- tutto a zero
        sequences_train=torch.cat((torch.zeros((batch.shape[0],1,self.configs.c_out)).to(batch.device),
                                   # sottrai successivi ai precedenti e trovi lo spostamento step-by-step
                                   batch[:,1:input_n,dim_used]-batch[:,:input_n-1,dim_used]),1)
        sequences_gt=batch[:,input_n:,dim_used]

        if self.configs.dec=="bayes":
            # passagli le velocità
            seq_run, kl_out = self.forward(sequences_train,self.configs.samples)
            seq = seq_run.mean(0)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:output_n-1,:]
                                # somma l'ultimo elemento dei train frames                            
            mpjpe = mpjpe_error(seq + batch[:,input_n-1:input_n,dim_used],sequences_gt)
            self.log_dict({'validation mpjpe':mpjpe,'kl':kl_out},on_step=True,on_epoch=True,prog_bar=True)
            return mpjpe+kl_out/self.configs.num_batches
        else:
            seq = self.forward(sequences_train,self.configs.samples)
            seq[:,1:,:]=seq[:,1:,:]+seq[:,:output_n-1,:]

                                # somma l'ultimo elemento dei train frames                            
            mpjpe = mpjpe_error(seq + batch[:,input_n-1:input_n,dim_used],sequences_gt)
            self.log_dict({'validation mpjpe':mpjpe},on_step=True,on_epoch=True,prog_bar=True)
            return mpjpe
