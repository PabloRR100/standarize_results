#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recivering results from previous experiments
"""

import os
import re
import glob
import json
import torch
import pickle
from results import *
from collections import OrderedDict as OD
from utils import model_Template as M
from utils import experiment_Template as E

path_models = './models'
path_results = './results'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
levels = ['Results_Single_Models.pkl', 'Results_Ensemble_Models.pkl', 'Results_Testing.pkl']

def collect(model, paths, small):
    
#    ls_results = list()
#    outputs = list()
    
    for p in paths:
    
        # Init Experiment
        e = E()
        
        # Paths to model weights
        regex = re.compile(r'{}'.format(small), re.IGNORECASE)
        ch = glob.glob(os.path.join(path_results, model, p, 'checkpoints/*.pkl'))
        ch_e = list(filter(regex.search, ch))
        ch_s = list(set(ch) - set(ch_e))
    
        # Load Single
        
        m = M()
        e.single = m
        
        l = levels[0]
        with open(os.path.join(path_results, model, p, l),'rb') as obj:
            r = pickle.load(obj)
#            ls_results.append(r)
            
        m.name = r.name
        m.best_acc = m.best_va_top1 = max(r.valid_accy)
        m.best_tr_top1 = max(r.train_accy) 
        
        m.tr_loss = r.train_loss
        m.tr_accy = r.train_accy
        m.va_loss = r.valid_loss
        m.va_accy = r.valid_accy
        
        m.model_weights = torch.load(ch_s[0], map_location=device)
#        outputs.append(m)
                
        # Load Ensemble 
        
        m = M()
        e.ensemble = m
        l = levels[1]
        with open(os.path.join(path_results, model, p, l),'rb') as obj:
            r = pickle.load(obj)
#            ls_results.append(r )
            
        m.name = r.name
        m.best_acc = m.best_va_top1 = max(r.valid_accy['ensemble'])
        m.best_tr_top1 = max(r.train_accy['ensemble'])
#        outputs.append(m)
        
        m.tr_loss = r.train_loss
        m.tr_accy = r.train_accy
        m.va_loss = r.valid_loss
        m.va_accy = r.valid_accy
        
        ensemble = OD()
        for n in range(r.m):
            ensemble['net_{}'.format(n)] = torch.load(ch_e[n], map_location=device)
        m.model_weights = ensemble
                s
        # Gather
        # ------
        e.name = p + ' vs ' + m.name.lower()
        with open(os.path.join(path_results, model, (e.name + '.pth')), 'wb') as j:
            pickle.dump(e, j, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path_results, model, (e.name + '.json')), 'wb') as j:
            json.dump(e.__tojson__())
            
        j = e.__tojson__()
#        import jsonpickle
#        json_object = jsonpickle.encode(e)
#        with open('data.json', 'w') as outfile:
#            json.dump(json_object, outfile, indent=4)


# ====
# VGGs
# ====

model = 'vggs'
small = 'vgg9'
paths = ['vgg13', 'vgg19']
collect(model, paths, small)

# =======
# ResNets
# =======

model = 'resnets'
paths = ['resnet56', 'resnet110']
checkpoints = ['ResNet56.pkl']
collect(model, paths)

# =========
# DenseNets
# =========

model = 'densenets'
paths = ['densenet121']
collect(model, paths)

# ==========
# Playground
# ==========

model = 'densenets'
paths = ['densenet121']
collect(model, paths)


            
with open('./results/resnets/resnet56 vs resnet20(x3).pkl', 'rb') as input:
    aaa = pickle.load(input)
            






