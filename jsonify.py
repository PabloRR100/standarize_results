"""
Collect the Results and dump JSONS of the trainings
"""

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
        self.model_wieghts = None
        # Full training results
#        self.tr_loss = None
#        self.tr_acc = None
#        self.va_loss = None
#        self.va_acc = None
        
   
def experiment_Template():
    
    def __init__(self):
        
        self.name = None
        self.single = None
        self.ensemble = None


def pickle_to_json():
    pass
