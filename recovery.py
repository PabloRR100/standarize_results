#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recivering results from previous experiments
"""

import os
import pickle
from results import *
from utils import model_Template as M
from utils import experiment_Template as E

path_models = './models'
path_results = './results'
levels = ['Results_Single_Models.pkl', 'Results_Ensemble_Models.pkl', 'Results_Testing.pkl']

def collect(model, paths):
    
    ls_results = list()
    outputs = list()
    
    for p in paths:
    
        e = E()
        
        m = M()
        m.name = p
        
        # Single  
        l = levels[0]
        with open(os.path.join(path_results, model, p, l),'rb') as obj:
            r = pickle.load(obj)
            ls_results.append(r)
            
        m.name = r.name
        m.best_acc = m.best_va_top1 = max(r.valid_accy)
        m.best_tr_top1 = max(r.train_accy) 
        outputs.append(m)
        e.single = m
            
        m = M()
        m.name = p
                
        # Ensemble 
        l = levels[1]
        with open(os.path.join(path_results, model, p, l),'rb') as obj:
            r = pickle.load(obj)
            ls_results.append(r)
            
        m.name = r.name
        m.best_acc = m.best_va_top1 = max(r.valid_accy['ensemble'])
        m.best_tr_top1 = max(r.train_accy['ensemble'])
        outputs.append(m)
        e.ensemble = m
                
        # Gather
        e.name = p + ' vs ' + m.name.lower()
        with open(os.path.join(path_results, model, (e.name + '.pkl')), 'wb') as j:
            pickle.dump(e, j, pickle.HIGHEST_PROTOCOL)



# ====
# VGGs
# ====

model = 'vggs'
paths = ['vgg13', 'vgg19']
collect(model, paths)

# =======
# ResNets
# =======

model = 'resnets'
paths = ['resnet56', 'resnet110']
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


            
with open('./results/resnet56 vs resnet20(x3).pkl', 'rb') as input:
    aaa = pickle.load(input)
            






