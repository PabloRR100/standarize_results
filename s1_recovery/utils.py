
import os
import time
import json
import torch
from collections import OrderedDict


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


# =============================================================================
# Load files to Google Drive
# =============================================================================

#from pydrive.auth import GoogleAuth
#from pydrive.drive import GoogleDrive
#
#gauth = GoogleAuth()
#gauth.LocalWebserverAuth()
#
#def creteFolder(name):
#    file_metadata = {'name': name, 
#                     'mimetype': 'application/vnd.google-apps.folder'}
#    file = drive_service.files().create(body=file_metadata, fields='id').execute()
#    print('Folder ID: %s' % file.get('id'))
#
#def uploadFile(filename, filepath, filetype):
#    
#    file_metadata = {'name': filename}
#    media = MediaFileUpload(filepath, mimetype=filetype)
#    file = drive_service.files().create(body=file_metadata,
#                                        media_body=media,
#                                        fields='id').execute()
#    print('File ID: %s' % file.get('id'))
#    return


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
    net.to(device)
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
