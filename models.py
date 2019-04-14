
import sys
sys.path.append('../Single_Ensembles/models')   # VGGs, Resnets
sys.path.append('../PyTorch_CIFAR/models')      # Densenets
sys.path.append('../Recursive_Networks/')       # Playground Architectures
sys.path.append('../Single_Ensembles/')       # Playground Architectures
from utils import count_parameters


from vggs import VGG
vgg9  = VGG('VGG9')
vgg13 = VGG('VGG13')
vgg19 = VGG('VGG19')

from resnets import ResNet20, ResNet56, ResNet110
resnet20 = ResNet20()
resnet56 = ResNet56()
resnet110 = ResNet110()

from densenet import DenseNet121, densenet_cifar
densenetCIF = densenet_cifar()
densenet121 = DenseNet121()

from scripts.models import Conv_Net
# net = Conv_Net



'''
VGGs
====
VGG13 vs VGG9 (x3)
VGG19 vs VGG9 (x7)


RESNETS
=======
RESNET56 vs RESNET20 (x3)
RESNET110 vs RESNET20 (x6)


DENSENETS
=========
DENSET121 vs DENSETCIFAR(x7)


PLAYGROUND
==========



'''

metrics = ['tr_accy', 'va_accy_top1', 'va_accy_top5', 'epoch_t', 'inf_t', ]

models = [{'vggs': 
            [{'single': vgg13, 'ensemble': (vgg9,3)},
            {'single': vgg19, 'ensemble': (vgg9,7)}],
        {'single': resnet56, 'ensemble': (resnet20,3)},
        {'single': resnet110, 'ensemble': (resnet20,6)},
        {'single': densenet121, 'ensemble': (densenetCIF,6)}
    ]


