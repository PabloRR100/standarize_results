#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Load Models from Previous Experiments
- Calculate Epoch Time
- Calculate Testset Inference Time  --> Would this depend on the batch size?
- Load Experiments
- Save metrics into corresponding experiments
"""

import os
import re
import sys
import glob
import time
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import dataloaders
sys.path.append(os.path.abspath('../models'))


# Paths
path_models = './models'

# Data
from train import run_epoch
from test import inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader, _ = dataloaders('CIFAR', 128, path='../../datasets')
criterion = nn.CrossEntropyLoss().cuda() if device == 'cuda' else  nn.CrossEntropyLoss()

## TODO: This ideally would be automated for every experiment name
def cuda(net):
    global device
    net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    return net

def measure_epoch(net, optimizer, criterion, epoch, trainloader, testloader, device):
        
    times = list()
    for _ in range(4):
        start = time.time()           
        run_epoch(net, optimizer, criterion, epoch, trainloader, testloader, device)       
        times.append(time.time() - start)
    return np.mean(times)

def measure_inference(net, optimizer, criterion, testloader, device):
        
    times = list()
    for _ in range(4):
        start = time.time()    
        inference(net, optimizer, criterion, testloader, device)
        times.append(time.time() - start)
    return np.mean(times)


# ====
# VGGs
# ====

from models import vgg9, vgg13, vgg19

print('\nVGG13 vs VGG9(x3)')

print('\nVGG13')
net = vgg13
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\nVGG9 x 3')
net = vgg9
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 3
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 3
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

#

print('\nVGG19 vs VGG9(x7)')

print('\nVGG19')
net = vgg19
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\nVGG9 x 7')
net = vgg9
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 7
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 7
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)



# =======
# ResNets
# =======

from models import resnet20, resnet56, resnet110, 

print('\nResnet56 vs Resnet20(x3)')

print('\nResnet56')
net = resnet56
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\nResnet20 x 3')
net = resnet20
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 3
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 3
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

#

print('\nResnet110 vs Resnet20(x6)')

print('\nResnet110')
net = resnet56
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\nResnet20 x 6')
net = resnet20
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 6
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 6
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)


# =========
# DenseNets
# =========

from models import densenet_cifar, densenet121

print('\nDensenet121 vs DensenetCifar(x6)')

print('\nDensenet121')
net = resnet56
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\nDensenetCifar x 6')
net = resnet20
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 6
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 6
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)


# ========================
# Playground Non Recursive
# ========================

from models import Conv_Net

single = {'L': 32, 'M': 64, 'BN': False} 

ensemb = [{'L': 32, 'M': 31, 'BN': False, 'K': 4 },   # Horizontal Division
          {'L': 32, 'M': 21, 'BN': False, 'K': 8 },
          {'L': 32, 'M': 17, 'BN': False, 'K': 12},
          {'L': 32, 'M': 14, 'BN': False, 'K': 16},
         
          {'L': 30, 'M': 32, 'BN': False, 'K': 4 },    # Vertical Division (M=32)
          {'L': 13, 'M': 32, 'BN': False, 'K': 8 },
          {'L':  8, 'M': 32, 'BN': False, 'K': 12},
          {'L':  5, 'M': 32, 'BN': False, 'K': 12},
         
          {'L': 12, 'M': 48, 'BN': False, 'K': 4 },    # Vertical Division (M=48)
          {'L':  4, 'M': 48, 'BN': False, 'K': 8 },
          {'L':  3, 'M': 48, 'BN': False, 'K': 12},
          {'L':  1, 'M': 48, 'BN': False, 'K': 16},
          
          {'L':  6, 'M': 64, 'BN': False, 'K': 4 },    # Vertical Division 
          {'L':  2, 'M': 64, 'BN': False, 'K': 8 },
          {'L':  1, 'M': 64, 'BN': False, 'K': 12}]


print('\nPlayground for L = 32, M = 64')
name = 'L = 32, M = 64'
net = Conv_Net()
net = cuda(net)
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, epoch, trainloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)


'L={}, M={}'


# ====================
# Playground Recursive
# ====================

from models import Conv_Recusive_Net