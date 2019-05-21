#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data import dataloaders
sys.path.append(os.path.abspath('../models'))

# Paths
path_models = './models'

# Data
from train import run_epoch
from infer import inference
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader, _ = dataloaders('CIFAR', 128, path='../../datasets')
criterion = nn.CrossEntropyLoss().cuda() if device == 'cuda' else  nn.CrossEntropyLoss()

## TODO: This ideally would be automated for every experiment name

def measure_epoch(net, optimizer, criterion, trainloader, testloader, device):
        
    times = list()
    for _ in range(4):
        start = time.time()           
        run_epoch(net, optimizer, criterion, trainloader, testloader, device)       
        times.append(time.time() - start)
    return np.mean(times)

def measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device):
        
    times = list()
    nets = [net] * K
    for _ in range(4):
        start = time.time()    
        for net in nets:
            run_epoch(net, optimizer, criterion, trainloader, testloader, device)       
        times.append(time.time() - start)
    return np.mean(times)

def measure_inference(net, optimizer, criterion, testloader, device):
    
    times = list()
    for _ in range(4):
        start = time.time()    
        inference(net, optimizer, criterion, testloader, device)
        times.append(time.time() - start)
    return np.mean(times)

def measure_inference_ensemble(net, K, optimizer, criterion, testloader, device):
        
    times = list()
    nets = [net] * K
    for _ in range(4):
        start = time.time()   
        for net in nets:
            inference(net, optimizer, criterion, testloader, device)
        times.append(time.time() - start)
    return np.mean(times)


# ====
# VGGs
# ====

from models import vgg9, vgg13, vgg19
print('\n\n\n\n VGG Architectures')

print('\n\n\nVGG13 vs VGG9(x3)')

print('\n\nVGG13')
net = vgg13
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\n\nVGG9 x 3')
K = 3
net = vgg9
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

#

print('\n\n\nVGG19 vs VGG9(x7)')

print('\n\nVGG19')
net = vgg19
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\n\nVGG9 x 7')
K = 7
net = vgg9
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)



# =======
# ResNets
# =======

from models import resnet20, resnet56, resnet110
print('\n\n\n\n Resnet Architectures')

print('\n\n\nResnet56 vs Resnet20(x3)')

print('\n\nResnet56')
net = resnet56
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\n\nResnet20 x 3')
K = 3
net = resnet20
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

#

print('\n\n\nResnet110 vs Resnet20(x6)')

print('\n\nResnet110')
net = resnet110
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\n\nResnet20 x 6')
K = 6
net = resnet20
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)


# =========
# DenseNets
# =========

from models import densenet_cifar, densenet121
print('\n\n\n\n Densenet Architectures')

print('\n\n\nDensenet121 vs DensenetCifar(x6)')

print('\n\nDensenet121')
net = densenet121
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device) * 1
inference_time = measure_inference(net, optimizer, criterion, testloader, device) * 1
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

print('\n\nDensenetCifar x 6')
K = 6
net = densenet_cifar
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)


# ========================
# Playground Non Recursive
# ========================

from models import Conv_Net
print('\n\n\n\n Playground Non Recursive Architectures')

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


print('\n\nPlayground Non Recursive for L = 32, M = 64')
name = 'L = 32, M = 64'
net = Conv_Net(name, single['L'], single['M'])
optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
epoch_time = measure_epoch(net, optimizer, criterion, trainloader, testloader, device)
inference_time = measure_inference(net, optimizer, criterion, testloader, device)
print('\nTraining Epoch Time : ', epoch_time)
print('\nTest set Inference Time : ', inference_time)

for ens in ensemb:
    
    ens = ensemb[0]
    L, M, _, K = ens.values()
    print('\n\nPlayground L={}, M={}, K={}'.format(L,M,K))
    
    name = 'L={}, M={}'.format(L,M)
    net = Conv_Net(name, L, M)
    optimizer = optim.SGD(net.parameters(), 0.01, momentum=0.9, weight_decay=1e-5)
    epoch_time = measure_epoch_ensemble(net, K, optimizer, criterion, trainloader, testloader, device)
    inference_time = measure_inference_ensemble(net, K, optimizer, criterion, testloader, device)
    print('\nTraining Epoch Time : ', epoch_time)
    print('\nTest set Inference Time : ', inference_time)


## ====================
## Playground Recursive
## ====================
#
#from models import Conv_Recusive_Net
#print('\n\n\n\n Playground Recursive Architectures')
#



