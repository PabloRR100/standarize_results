# Results


## Organization

### Folders
- `models`                  # Code of the models used (vggs, resnets, densenets and playgrounds)  
- `results`                 # Collection of the results obtained on the different repos for the each of the models  
 | -  `network name`        # e.g. vggs / resnets / densenets / playground  
    | - `net model`         # e.g. vgg13 / resnet56 / densenet121 / m64_l32  
      | - `checkpoints`     # model weights  
      | - `dicts`           # training results stored as pickle objects (Train_Results python class)  
      | - `logs`            # .txt files with the information of the training  
- `experiments`             # This folder is only on Drive since the files are too big -> run `s1_recovery.py` to populate it  
  | - `network name`        # vggs / resnets / densenets / playground  
    | - `experiment.json`   # single_name vs individual_of_ensemble(x ensemble_size).json  
    | - `experiment.pth`    # single_name vs individual_of_ensemble(x ensemble_size).pth  
  
  
### Files

- `data.py`: helpers to load the data  
- `results.py`: class used to store results during training in deprecated repositories  
- `s1_recovery.py`: script to gather result from previous experiments and populate `experiment` folder  
- `templates.py`: class used to store results of the experiments  
- `utils.py`: helpers  



## Objectives

In this repo I have included script with 2 purposes:  
  
### 1 - Standarization  
The different experiments have been run using different code and different ways of storing the results.  
We are using script s1_recovery to extract all of the results and convert them into the same "trusted" format.  

Current Format  
--------------  
  
- VGGs and Resnets  
For VGGs and ResNets we have been saving the .pkl objects as the checkpoints in different files for the singlemodel and each of the individual models of the ensemble.

- Densenets  
We have been using checkpoints storing acc, epoch and the state dict of the models:
    - For the single just `checkpoint['net']`  
    - For the ensemble -> `checkpoint['net_i']` where i starts on 0 up to the number of networks on the ensemble  
    
- Playground  
We have used again checkpoints with the unique name L_{}_M_{}_K_{} to store the models.  
For the results of training we have the prefix Single_ or Ensemble_ with above ids.  


Trusted Format  
--------------  

We define a trusted structured to store every experiments with the same format.  
This allows easier exploration of the results and the possibility to use the same plotting script for every model.  

This template has been used to store the results of a model:  
```python
class model_Template():
    
    def __init__(self):
        
        # Training Summary
        self.name = None
        self.best_acc_epoch = None
        self.best_tr_top1 = None
        self.best_va_top1 = None
        self.tr_epoch_time = None
        self.testset_inf_time = None
        
        #  Full training results
        self.tr_loss = None
        self.tr_accy = None
        self.va_loss = None
        self.va_accy = None
        
        # Model Weights
        self.model_weights = None
        
```

And the following template is for an experiment, where we compare a single model with an ensemble of models of the same budget.  
```python
class experiment_Template():
    
    def __init__(self):
        
        self.name = None        # Name of the experiment
        self.single = None      # Single Model Populted Template
        self.ensemble = None    # Esemble Model Populated Template (tr_loss and so on will be OrderedDicts of each of the individuals) 
```

### 2 - Reproducibility

