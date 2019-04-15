#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recivering results from previous experiments
"""

import os
import glob
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
    
        # Single  
        
        m = M()
        e.single = m
        
        l = levels[0]
        with open(os.path.join(path_results, model, p, l),'rb') as obj:
            r = pickle.load(obj)
#            ls_results.append(r)

        ch = glob.glob(os.path.join(path_results, model, p, 'checkpoints/*.pkl'))
        weights = pickle.load(g)
            
        m.name = r.name
        m.best_acc = m.best_va_top1 = max(r.valid_accy)
        m.best_tr_top1 = max(r.train_accy) 
        m.model_weights = None
        m.tr_loss = r.train_loss
        m.tr_acc = r.train_accy
        m.va_loss = r.valid_loss
        m.va_acc = r.valid_acc
#        outputs.append(m)
                
        # Ensemble 
        
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
        m.model_weights = None
        m.tr_loss = r.train_loss
        m.tr_acc = r.train_accy
        m.va_loss = r.valid_loss
        m.va_acc = r.valid_accy
                
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
            






