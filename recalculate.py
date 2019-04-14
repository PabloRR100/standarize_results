
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

path = paths[0]
check_paths = glob.glob(os.path.join(path_results, model, path, 'checkpoints/*.pkl'))

# Single
check_path = [c for c in check_paths if models[0] in c][0]
net = ResNet56()
net = load_model_single(net, check_path, device)



    
print('Loading Model')
checkpoint = torch.load(check_path, map_location=device)


check_paths = [c for c in check_paths if small in c]
net = ResNet20()
#for c in check_paths:

# Ensemble

def load_weights_ensemble(check_path):
    assert os.path.exists(check_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(check_path, map_location=device)
    new_state_dict = OrderedDict()
    
    for k,v in checkpoint['net_{}'.format(n)].state_dict().items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    return new_state_dict 
    
def load_model_ensemble(net, check_path, device):    
    net.load_state_dict(load_weights(check_path)) # remove word `module`
    net.to(device)
    if device == 'cuda': 
        net = torch.nn.DataParallel(net)
    return net
    

for n,net in enumerate(ensemble.values()):
    net = load_model(net, n+1, check_path, device)
    


# ===================
#  Training Top1 Top5 
# ===================
    
    
# =====================
#  Validation Top1 Top5 
# =====================
    
    
# ================
# Train Epoch Time
# ================
    
    
# ====================
# Valid Inference Time
# ====================
    
    
