# Results

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
        self.best_tr_top5 = None
        self.best_va_top1 = None
        self.best_va_top5 = None
        self.tr_epoch_time = None
        self.testset_inf_time = None
        
        #  Full training results
        self.tr_loss = None
        self.tr_accy = None
        self.va_loss = None
        self.va_accy = None
        
        # Model Weights
        self.model_weights = None
        
    def __repr__(self):
        
        printable = ['name', 'best_acc', 'tr_epoch_time', 'testset_inf_time']
        attrs = vars(self)
        attrs = {k:v for k,v in attrs.items() if k in printable}
        return ', '.join("\n%s: %s" % item for item in attrs.items())
    
    def __json__(self):
        
        return dict(
            name = self.name,
            best_acc = self.best_va_top1,
            best_acc_epoch = self.best_acc_epoch,
            train_epoch_time = self.tr_epoch_time,
            test_set_inference_time = self.testset_inf_time,
            tr_loss = NoIndent(self.tr_loss), tr_accy = NoIndent(self.tr_accy),
            va_loss = NoIndent(self.va_loss), va_accy = NoIndent(self.va_accy))
```

And the following template is for an experiment, where we compare a single model with an ensemble of models of the same budget.  
```python
class experiment_Template():
    
    def __init__(self):
        
        self.name = None
        self.single = None
        self.ensemble = None
        
    def __repr__(self):
        
        attrs = vars(self)
        return ', '.join("\n\n%s: %s" % item for item in attrs.items())
    
    def __tojson__(self):
        
        return dict(name = self.name,
                    single = js(self.single.__json__()), 
                    ensemble = js(self.ensemble.__json__()))
```

### 2 - 

