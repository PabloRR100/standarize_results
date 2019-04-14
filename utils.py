
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
    
    def load_weights_single(check_path):
        assert os.path.exists(check_path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(check_path, map_location=device)
        
        new_state_dict = OrderedDict()
        for k,v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        return new_state_dict 

    net.load_state_dict(load_weights_single(check_path)) # remove word `module`
    net.to(device)
    if device == 'cuda': 
        net = torch.nn.DataParallel(net)
    return net