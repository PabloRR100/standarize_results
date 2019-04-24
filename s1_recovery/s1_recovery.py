#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- Collecting results from previous experiments in different formats
- Converting results to a 'trusted structure'
- Saving results into shared folder on Drive
"""

import os
import re
import glob
import json
import torch
import pickle
import numpy as np
from results import *
from collections import OrderedDict as OD
from templates import model_Template as M
from templates import experiment_Template as E
from templates import MyEncoder

path_models = './models'
path_results = './results'
path_experiments = '../experiments'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
levels = {'state_of_art': {'single': 'Results_Single_Models.pkl', 
                           'ensemble': 'Results_Ensemble_Models.pkl'},
          'playground': {'single': 'Single_', 
                         'ensemble': 'Ensemble_'}}

'''

# ============
# STATE OF ART
# ============

# VGGs and Resnets
# ================
For VGGs and ResNets we have been saving the .pkl objects of the checkpoints
in different files for the singlemodel and each of the individual models of the ensemble.

# DensenNet 
# =========
We have been using checkpoints storing acc, epoch and the state dict of the models:
    For the single just checkpoint['net']
    For the ensemble -> checkpoint['net_i'] where i starts on 0
    
    
# ========================
# Playground Architectures
# ========================
    
We have used again checkpoints woith L_{}_M_{}_K_{} to store the models.
For the results of training we have the prefix Single_ or Ensemble_ with above ids

    (*) Big difference:
        SoA experiments compares 1 singles model vs 1 ensemble of models
        Playground experiments compares 1 single model vs many ensembles of models
    

'''

    

def collect(model, paths, small):

    global levels
    for p in paths:
    
        # Init Experiment
        
        e = E()
        
        if model == 'vggs' or model == 'resnets':
            # Path to dicts
            dict_s = os.path.join(path_results, model, p, 'dicts', levels['state_of_art']['single'])
            dict_e = os.path.join(path_results, model, p, 'dicts', levels['state_of_art']['ensemble'])

            # Paths to model weights
            regex = re.compile(r'{}'.format(small), re.IGNORECASE)
            ch = glob.glob(os.path.join(path_results, model, p, 'checkpoints/*.pkl'))  ## This changes for densenets and playground
            ch_e = list(filter(regex.search, ch))
            ch_s = list(set(ch) - set(ch_e))
            # Loading weights
            single_weights = torch.load(ch_s[0], map_location=device)
            ensemble_weights = OD()
            for n,c in enumerate(ch_e):
                ensemble_weights['net_{}'.format(n)] = torch.load(c, map_location=device)
                
        elif model == 'densenets':
            # Path to dicts
            dict_s = os.path.join(path_results, model, p, 'dicts', levels['state_of_art']['single'])
            dict_e = os.path.join(path_results, model, p, 'dicts', levels['state_of_art']['ensemble'])
            # Paths to model weigths
            ch_s = glob.glob(os.path.join(path_results, model, p, 'checkpoints/*single*'))[0]
            ch_e = glob.glob(os.path.join(path_results, model, p, 'checkpoints/*ensemble*'))[0]
            # Loading weights
            single_weights = torch.load(ch_s, map_location=device)
            single_weights = {k:v for k, v in single_weights.items() if 'net' in k} 
            ensemble_weights = torch.load(ch_e, map_location=device)
            ensemble_weights = {k:v for k, v in ensemble_weights.items() if 'net' in k} 
        
        elif model == 'playground':
            # Path to dicts
            dict_s = glob.glob(os.path.join(path_results, model, small, 'dicts/*single*'))[0]
            dict_e = glob.glob(os.path.join(path_results, model, small, 'dicts/*ensemble*'))
            # Paths to model weights
            ch_s = glob.glob(os.path.join(path_results, model, small, 'checkpoints/*single*'))[0]
            ch_e = glob.glob(os.path.join(path_results, model, small, 'checkpoints/*ensemble*'))
            
            # Loading weights
            single_weights = torch.load(ch_s, map_location=device)
            single_weights = {k:v for k, v in single_weights.items() if 'net' in k} 
            
            ensemble_weights = list()
            for c in ch_e:
                weights = torch.load(c, map_location=device)
                ensemble_weights = {k:v for k, v in ensemble_weights.items() if 'net' in k} 
                
        else:
            print('[ERROR]: Model name not found')
        
        # Load Single
        
        m = M()
        e.single = m

#        l = levels[0]        
#        with open(os.path.join(path_results, model, p, 'dicts', l),'rb') as obj:
        with open(dict_s,'rb') as obj:
            r = pickle.load(obj)
        
        m.name = r.name
        m.best_acc_epoch = int(np.argmax(r.valid_accy))
        m.best_va_top1 = max(r.valid_accy)
        m.best_tr_top1 = max(r.train_accy) 
        
        m.tr_loss = r.train_loss
        m.tr_accy = r.train_accy
        m.va_loss = r.valid_loss
        m.va_accy = r.valid_accy
        
        m.model_weights = single_weights
                
        # Load Ensemble 
        
        m = M()
        e.ensemble = m
#        l = levels[1]
#        with open(os.path.join(path_results, model, p, 'dicts', l),'rb') as obj:
        with open(dict_e,'rb') as obj:
            r = pickle.load(obj)
            
        m.name = r.name
        m.best_acc_epoch = int(np.argmax(r.valid_accy['ensemble']))
        m.best_va_top1 = max(r.valid_accy['ensemble'])
        m.best_tr_top1 = max(r.train_accy['ensemble'])
        
        m.tr_loss = r.train_loss
        m.tr_accy = r.train_accy
        m.va_loss = r.valid_loss
        m.va_accy = r.valid_accy
        
        m.model_weights = ensemble_weights
                
        # Gather
        # ------
        e.name = p + ' vs ' + m.name.lower()
        j = e.__tojson__()
        j = j['name'] + j['single'] + j['ensemble']
        
        # Save as pickle object
        with open(os.path.join(path_experiments, model, (e.name + '.pth')), 'wb') as f:
            pickle.dump(e, f, pickle.HIGHEST_PROTOCOL)
        
        # Save as JSON file
        with open(os.path.join(path_experiments, model, (e.name + '.json')), 'w') as f:
            json.dump(j,f, cls=MyEncoder)
        

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


