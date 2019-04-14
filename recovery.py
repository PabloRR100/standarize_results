#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recivering results from previous experiments
"""

import os
import sys
import pickle
from results import *
from jsonify import model_Template as M
from jsonify import experiment_Template as E

# =======
# ResNets
# =======

# Collect results from saved objects

path_models = '../Single_Ensembles/models'
path_results = '../Single_Ensembles/results'

sys.path.append(path_models)   
from resnets import ResNet20, ResNet56, ResNet110
resnet20 = ResNet20()
resnet56 = ResNet56()
resnet110 = ResNet110()

paths = ['ResNet56', 'ResNet110']
levels = ['Results_Single_Models.pkl', 'Results_Ensemble_Models.pkl', 'Results_Testing.pkl']

ls_results = list()
outputs = list()

for p in paths:

    e = E
    e.name = p + ' vs ' 
    
    for l in levels:
    
        with open(os.path.join(path_results, 'dicts/resnets/definitives', p, l),'rb') as obj:
            r = pickle.load(obj)
            ls_results.append(r)
            
            # Single  
            m = M
            m.name = p
            m.best_acc = m.best_va_top1 = max(r.valid_acc)
            m.best_tr_top1 = max(r.train_acc)
            
            # Ensemble 
            
            
            


# Load Model and Calculate remaining metrics


