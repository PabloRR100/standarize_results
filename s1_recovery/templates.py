
import re
import json
from _ctypes import PyObj_FromPtr

# =========
# TEMPLATES
# =========


def js(d:dict):
    return json.dumps(d, cls=MyEncoder, indent=4)


class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        self.value = value

class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))

    def __init__(self, **kwargs):
        # Save copy of any keyword argument values needed for use here.
        self.__sort_keys = kwargs.get('sort_keys', None)
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                else super(MyEncoder, self).default(obj))

    def encode(self, obj):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.
        json_repr = super(MyEncoder, self).encode(obj)  # Default JSON.

        # Replace any marked-up object ids in the JSON repr with the
        # value returned from the json.dumps() of the corresponding
        # wrapped Python object.
        for match in self.regex.finditer(json_repr):
            # see https://stackoverflow.com/a/15012814/355230
            id = int(match.group(1))
            no_indent = PyObj_FromPtr(id)
            json_obj_repr = json.dumps(no_indent.value, sort_keys=self.__sort_keys)

            # Replace the matched id string with json formatted representation
            # of the corresponding Python object.
            json_repr = json_repr.replace(
                            '"{}"'.format(format_spec.format(id)), json_obj_repr)

        return json_repr

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

#i = json.dumps(e.single.__json__(), indent=4)
#j = json.dumps(e.ensemble.__json__(), indent=4)
#j = json.dumps(mod.__json__(), indent=4)

