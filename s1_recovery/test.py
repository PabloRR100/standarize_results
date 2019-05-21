#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from utils import timeit
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


@timeit
def inference(net, optimizer, criterion, epoch, dataloader, device):
    
    net.eval()
    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets) 
            del loss
            
    return
