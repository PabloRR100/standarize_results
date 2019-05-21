# -*- coding: utf-8 -*-

import os
import torch
from models import *
from collections import OrderedDict as OD

class Ensemble:
        
    '''
    Ensemble of deep neural networks ready for inference
    '''
    def __init__(self, net:list, size:int=None, method='average'):
        super(Ensemble).__init__()
        
        methods = ['average', 'soft_voting', 'mayority_voting']
        method_err= 'Method should belong to: {}'.format(methods)
        assert method in methods, method_err
        
        if size is None:
            assert isinstance(net, list), \
            'Models should be a list if size is not provided'
            self.nets = net
            self.size = len(net)
            
        else:
            assert not isinstance(net, list), \
            'If size is provide, pass just a single Model'
            self.net = [net('n{}'.format(n) for n in range(size))]
            self.size = size
            
    def train(self):
        for net in self.nets:
            net.train()
                
    def eval(self):
        for net in self.nets:
            net.eval()
            
    def forward(self, x, device):
        '''
        :Input: Tensor
        :Output: List of predictions for each model and the ensemble'
        '''
        outputs = list()
        for n, net in enumerate(self.nets):
            net.to(device)
            outputs.append(net(x))
        output = torch.mean(torch.stack(outputs), dim=0)
        return output
    
    def load(self, check_path, device):
        
        assert os.path.exists(check_path), 'Error: no checkpoint directory found!'    
        checkpoint = torch.load(check_path, map_location=device)
    
        for n,net in enumerate(self.nets):
            
            # DataParallel Agnostic --> Remove module. from the keys
            new_state_dict = OD()
            for k,v in checkpoint['net_{}'.format(n)].items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            # PyTorch version Agnostic --> Remove extra layers that we don't have
            model_dict = net.state_dict()
            pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
            
            model_dict.update(pretrained_dict) 
            net.load_state_dict(model_dict) 



if '__name__' == '__main__':
    
        
    net1 = Conv_Net('n1', 32, 64)
    net2 = Conv_Net('n2', 32, 64)
    net3 = Conv_Net('n3', 32, 64)
    nets = [net1, net2, net3]        
    
    ensemble1 = Ensemble(nets)
    ensemble2 = Ensemble(Conv_Net, 3)
    ensembles = [ensemble1, ensemble2]
    
    images, labels = next(iter(trainloader))
    images, labels = images.to(device), labels.to(device)
