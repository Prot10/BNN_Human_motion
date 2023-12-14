from funcs import dataloader as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
from funcs import loss
from .utils.data_utils import define_actions
from .visualizations import visualize
import time
from .dataloader import load_dataset

import torch.nn.functional as F

import os

# Use GPU if available, otherwise stick with cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

trainset, valset = load_dataset()

# # Arguments to setup the datasets
# datas = 'h36m' # dataset name
# path = './data/h3.6m/h3.6m/dataset'
# input_n=10 # number of frames to train on (default=10)
# output_n=25 # number of frames to predict on
# input_dim=3 # dimensions of the input coordinates(default=3)
# skip_rate=1 # # skip rate of frames
# joints_to_consider=22

# # Load Data
# print('Loading Train Dataset...')
# dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=0)
# print('Loading Validation Dataset...')
# vald_dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=1)

# batch_size=256

# print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
# data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)#

# print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
# vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
