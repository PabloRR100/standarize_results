#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Load Models from Previous Experiments
- Calculate Epoch Time
- Calculate Testset Inference Time  --> Would this depend on the batch size?
"""

import os
import re
import glob
import torch
import pickle
import numpy as np
from results import *
from data import dataloaders
from collections import OrderedDict as OD
from templates import model_Template as M
from templates import experiment_Template as E
from templates import MyEncoder

import sys
sys.path.append(os.path.abspath('../models'))
sys.path

# Data
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader, classes = dataloaders('CIFAR', 128)

## TODO: Problem Loading Weights
import sys
sys.path.append(os.path.abspath('../models'))
from models import *
from models.resnets import *


path_models = './models'
path_results = './results'
path_experiments = '../experiments'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
levels = {'state_of_art': {'single': 'Results_Single_Models.pkl', 
                           'ensemble': 'Results_Ensemble_Models.pkl'},
          'playground': {'single': 'Single_', 
                         'ensemble': 'Ensemble_'}}
    

# Time to inference entire Test Set



        

# ====
# VGGs
# ====

model = 'vggs'
small = 'vgg9'
paths = ['vgg13', 'vgg19']
#collect(model, paths, small)

# =======
# ResNets
# =======

model = 'resnets'
small = 'resnet20'
paths = ['resnet56', 'resnet110']
#collect(model, paths, small)

# =========
# DenseNets
# =========

model = 'densenets'
small = 'densenet_cifar'
paths = ['densenet121']
#collect(model, paths, small)

# ==========
# Playground
# ==========

# CANDIDATES
model = 'playground'
single = {'L': 32, 'M': 64, 'BN': False} 
small = 'm{M}_l{L}'.format(**single)

#paths = [single]
paths = [{'L': 32, 'M': 31, 'BN': False, 'K': 4},   # Horizontal Division
         {'L': 32, 'M': 21, 'BN': False, 'K': 8},
         {'L': 32, 'M': 17, 'BN': False, 'K': 12},
         {'L': 32, 'M': 14, 'BN': False, 'K': 16},
         
         {'L': 6, 'M': 64, 'BN': False, 'K': 4},    # Vertical Division
         {'L': 2, 'M': 64, 'BN': False, 'K': 8},
         {'L': 1, 'M': 64, 'BN': False, 'K': 12},]

collect(model, paths, small)


#with open('./results/playground/m64_l32/dicts/Single_Non_Recursive_L_32_M_64_BN_False.pkl', 'rb') as inp:
#    result = pickle.load(inp)
##    result.name = 'L_32_M_64_BN_False'
#    aa = [result]
#
#with open('./results/playground/m64_l32/dicts/Single_Non_Recursive_L_32_M_64_BN_False.pkl', 'wb') as f:
#            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

result = torch.load('./results\playground\m64_l32\checkpoints\Ensemble_Non_Recursive_L_13_M_32_BN_False_K_8.t7', map_location='cpu')
aa = [result]


## Round 1
#ensemble = [{'L': 16, 'M': 31, 'BN': False, 'K': 4} ,
#            {'L': 4,  'M': 36, 'BN': False, 'K': 16},
#            {'L': 4, 'M': 54, 'BN': False, 'K': 8},
#            {'L': 8, 'M': 40, 'BN': False, 'K': 8}]
#
## Round 2
#ensemble = [{'L': 16, 'M': 31, 'BN': False, 'K': 4} ,
#            {'L': 4,  'M': 36, 'BN': False, 'K': 16},
#            {'L': 4, 'M': 54, 'BN': False, 'K': 8},
#            {'L': 8, 'M': 40, 'BN': False, 'K': 8}]
#
#ensemble = [{'L': 32, 'M': 31, 'BN': False, 'K': 4},   # Horizontal Division
#            {'L': 32, 'M': 21, 'BN': False, 'K': 8},
#            {'L': 32, 'M': 17, 'BN': False, 'K': 12},
#            {'L': 32, 'M': 14, 'BN': False, 'K': 16},
#            
#            {'L': 6, 'M': 64, 'BN': False, 'K': 4},    # Vertical Division
#            {'L': 2, 'M': 64, 'BN': False, 'K': 8},
#            {'L': 1, 'M': 64, 'BN': False, 'K': 12},]
#
## Round 3
#ensemble = [{'L': 12,  'M': 48, 'BN': False, 'K': 4},       # 3.2 M = 32
#            {'L': 5,   'M': 48, 'BN': False, 'K': 8},
#            {'L': 3,   'M': 48, 'BN': False, 'K': 12},
#            {'L': 1,   'M': 48, 'BN': False, 'K': 16},
#            {'L': 30,  'M': 32, 'BN': False, 'K': 4},       # 3.2 M = 32
#            {'L': 13,  'M': 32, 'BN': False, 'K': 8},    
#            {'L': 8,   'M': 32, 'BN': False, 'K': 12},     
#            {'L': 5,   'M': 32, 'BN': False, 'K': 16}]    


