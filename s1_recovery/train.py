#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from utils import timeit
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(net, optimizer, criterion, dataloader, device):
    
    net.train()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        optimizer.zero_grad()

        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        del predicted

    return


def test(net, optimizer, criterion, dataloader, device):
    
    net.eval()
    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets) 
            del loss
            
    return


#@timeit
def run_epoch(net, optimizer, criterion, trainloader, testloader, device):
    
    net.to(device)
    train(net, optimizer, criterion, trainloader, device)
    test(net, optimizer, criterion, testloader, device)

