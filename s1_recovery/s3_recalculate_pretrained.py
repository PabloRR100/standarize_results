
'''

1 - Load saved experiment from s1_recovery.py
2 - Load model weights from that template
3 - Recalculate time metrics (possibly missing accuracy metrics)
4 - Save experiment with the new metrics addded

'''


import os
import glob
import torch
import pickle
from results import *
from data import dataloaders
from collections import OrderedDict
from utils import model_Template as M
from utils import experiment_Template as E

device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainloader, testloader, classes = dataloaders('CIFAR', 128)


# =============================================================================
# RESNETS
# =============================================================================

from models.resnets import ResNet20, ResNet56, ResNet110

path_results = './results'
model = 'resnets'
small = 'ResNet20'
paths = ['resnet56', 'resnet110']
models = ['ResNet56', 'ResNet110']


# ================
# Load the weights
# ================

from utils import load_model_single
from utils import single_test_accuracies

path = paths[0]
check_paths = glob.glob(os.path.join(path_results, model, path, 'checkpoints/*.pkl'))

with open('./results/resnets/resnet56 vs resnet20(x3).pkl', 'rb') as input:
    experiment = pickle.load(input)

# Single
print('\nLoading Trained Model')
check_path = [c for c in check_paths if models[0] in c][0]
net = ResNet56()
net = load_model_single(net, check_path, device)
print('\nCalculating Training Stats')
tr_top1, tr_top5, _ = single_test_accuracies(net, trainloader, device)
print('\nCalculating Validation Stats')
va_top1, va_top5, va_time = single_test_accuracies(net, testloader, device)

print('\nSaving changes into Experiment')
experiment.singlebest_tr_top5 = va_top1
experiment.singlebest_va_top5 = va_top5
experiment.singletestset_inf_time = va_time
experiment.single.model_weights = net.state_dict()

with open('./results/resnets/resnet56 vs resnet20(x3).t7', 'wb') as obj:
    experiment = pickle.dump(experiment, obj, pickle.HIGHEST_PROTOCOL)

print('\nExiting')
exit()

#state_single = experiment.single
#state_ensemble = experiment.ensemble
#aaa = [state_single]




#
check_paths = [c for c in check_paths if small in c]
net = ResNet20()
#for c in check_paths:

# Ensemble

def load_weights_ensemble(check_path):
    assert os.path.exists(check_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(check_path, map_location=device)
    new_state_dict = OrderedDict()
    
    # Remove module. in case training was done using Parallelization
    for k,v in checkpoint['net_{}'.format(n)].state_dict().items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict 

    # Remove unnecesary keys in case model was trained in a different PyTorch version
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    net.load_state_dict(pretrained_dict)
    
    
def load_model_ensemble(net, check_path, device):    
    net.load_state_dict(load_weights(check_path)) # remove word `module`
    net.to(device)
    if device == 'cuda': 
        net = torch.nn.DataParallel(net)
    return net
    
for n,net in enumerate(ensemble.values()):
    net = load_model(net, n+1, check_path, device)
    

#
## ===================
##  Training Top1 Top5 
## ===================
#    
#    
## =====================
##  Validation Top1 Top5 
## =====================
#    
#    
## ================
## Train Epoch Time
## ================
#    
#    
## ====================
## Valid Inference Time
## ====================
#    
#    
