import torch.nn as nn
from .pos_embed import Pos_Embed
from .encoder import *
from .decoder import *

import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from ..loss import mpjpe_error


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initializes conv layers using He initialization
def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')

    

# Initializes fc layers using He initialization    
def fc_init(fc):
    nn.init.kaiming_normal_(fc.weight, nonlinearity='relu')

    

# Initializes bn layers by setting the weights to a given scale and biases to zero
def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)
    


class Model(nn.Module):

    def __init__(self, num_channels, num_frames_out,
                 old_frames, num_joints, num_heads, drop,
                 d_model=512, num_predictions=10,
                 return_sample=False, config=None):
        super().__init__()

        # Configuration for the model
        if config==None:
            self.config = [[d_model, 10, 8], [d_model, 8, 6], [d_model, 6, 4],
                           [d_model, 4, 2], [d_model, 2, 1]]

        # Assigning parameters to the class instance
        self.num_channels = num_channels
        self.num_frames_out = num_frames_out
        self.num_heads = num_heads
        self.num_joints = num_joints
        self.old_frames = old_frames
        self.d_model = d_model
        self.num_predictions = num_predictions
        self.return_sample = return_sample

        # Linear and normalization layers for processing inputs
        self.lin = nn.Sequential(nn.Linear(self.num_channels*self.num_joints, d_model),nn.BatchNorm1d(self.old_frames))
        self.norm = nn.BatchNorm2d(self.num_channels)

        # List of EncoderBlocks
        self.blocks = nn.ModuleList()
        for index, (d_, in_, out_) in enumerate(self.config):
            self.blocks.append(EncoderBlock(num_heads=self.num_heads,
                                            d_model=d_, time_in=in_, time_out=out_,
                                            num_joints=self.num_joints, dropout=drop))
        # Positional embedding layer
        self.pos = Pos_Embed(self.num_channels,self.old_frames,self.num_joints)
        
        # Decoder instance
        self.dec = Decoder(self.d_model, num_predictions)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m,1)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m,1)
            elif isinstance(m, nn.Linear):
                fc_init(m)


    def forward(self, x, num_predictions):
        
        # Initial tensor processing
        x = x.view(-1, self.old_frames, self.num_joints, self.num_channels).permute(0, 3, 1, 2)
        x = (x + self.pos(x)).permute(0, 2, 3, 1).view(-1, self.old_frames, self.num_joints * self.num_channels)
        x = self.lin(x)

        # Sequentially passing the input through each EncoderBlock
        for i, block in enumerate(self.blocks):
            x = block(x)

        # Preparing the context for the decoder
        context = x.view(-1, self.d_model).unsqueeze(0)

        # Initialize VARIABLES to store the results
        all_results = []
        kl_loss = 0

        # Repeat the decoding process num_predictions times
        for _ in range(num_predictions):
            results, kl_loss_step = self.dec(hidden=context, num_steps=self.num_frames_out)
            all_results.append(results)
            kl_loss += kl_loss_step

        # Converting list to tensor
        all_results = torch.stack(all_results)
        all_results = all_results.view(num_predictions, all_results.shape[1], all_results.shape[2], 22, 3)
        
        # Compute the average of results and KL losses
        avg_results = torch.mean(all_results.view(num_predictions, all_results.shape[1], all_results.shape[2], 22, 3), dim=0)
        avg_kl_loss = kl_loss / num_predictions

        # Return all the sample if return_sample is True
        if self.return_sample:
            return all_results, avg_results.view(all_results.shape[1], all_results.shape[2], 66).permute(1, 0, 2), avg_kl_loss

        return avg_results.view(all_results.shape[1], all_results.shape[2], 66).permute(1, 0, 2), avg_kl_loss
    
    
    
    def count_parameters(self):

        # Counts the number of  parameters of the model
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        return trainable_params, non_trainable_params, total_params