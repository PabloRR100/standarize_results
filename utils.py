
import os
import torch
from collections import OrderedDict

# =========
# TEMPLATES
# =========

class model_Template():
    
    def __init__(self):
        
        self.name = None
        self.best_acc = None
        self.best_tr_top1 = None
        self.best_tr_top5 = None
        self.best_va_top1 = None
        self.best_va_top5 = None
        self.tr_epoch_time = None
        self.testset_inf_time = None
        self.model_weights = None
        # Full training results
#        self.tr_loss = None
#        self.tr_acc = None
#        self.va_loss = None
#        self.va_acc = None
        
   
class experiment_Template():
    
    def __init__(self):
        
        self.name = None
        self.single = None
        self.ensemble = None


def pickle_to_json():
    pass


# =============================================================================
# Load Models
# =============================================================================
 
    
def load_model_single(net, check_path, device):   
    
    assert os.path.exists(check_path), 'Error: no checkpoint directory found!'    
    checkpoint = torch.load(check_path, map_location=device)
    
    # DataParallel Agnostic --> Remove module. from the keys
    new_state_dict = OrderedDict()
    for k,v in checkpoint.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    # PyTorch version Agnostic --> Remove extra layers that we don't have
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    
    model_dict.update(pretrained_dict) 
    net.load_state_dict(model_dict) ##
    return net





# =============================================================================
# METRICS
# =============================================================================

import time
import numpy as np
from results import accuracies

## TEST TOP-K ACCURACY
# --------------------

#def validset_time(net, testloader, device):
#    start = time.time()
#    net.eval()
#    with torch.no_grad():
#        for images, labels in testloader:
#            images, labels = images.to(device), labels.to(device)
#            outputs = net(images)            
#            _,_ = torch.max(outputs.data, 1)
#    return time.time() - start            


def single_test_accuracies(net, testloader, device):
           
    start = time.time()
    
    net.eval()
    prec1, prec5 = list(), list()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # Ensemble forward pass
            outputs = net(images)            
            _, predicted = torch.max(outputs.data, 1)
    
            # General Results Top1, Top5
            p1, p5 = accuracies(outputs.data, labels.data, topk=(1, 5))
            prec1.append(p1.item())
            prec5.append(p5.item())
    
    elapsed = time.time() - start            
    print('Top-1 Accuracy = ', np.mean(prec1))
    print('Top-5 Accuracy = ', np.mean(prec5))
    print('Full validset proccesed in ', elapsed)
    return round(np.mean(prec1),3), round(np.mean(prec5),3), elapsed

def ensemble_test_accuracies(ensemble, testloader, device):

    prec1, prec5 = list(), list()    
    for net in ensemble.values():
        net.eval()
        
    with torch.no_grad():        
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # Ensemble forward pass
            individual_outputs = list()
            for net in ensemble.values():
                outputs = net(images)
                individual_outputs.append(outputs)
                
            outputs = torch.mean(torch.stack(individual_outputs), dim=0)
            _, predicted = torch.max(outputs.data, 1)
    
            # General Results Top1, Top5
            p1, p5 = accuracies(outputs.data, labels.data, topk=(1, 5))
            prec1.append(p1.item())
            prec5.append(p5.item())
    
    print('Top-1 Accuracy = ', np.mean(prec1))
    print('Top-5 Accuracy = ', np.mean(prec5))
    
    return round(np.mean(prec1),3), round(np.mean(prec5),3)
