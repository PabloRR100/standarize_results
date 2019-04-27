
import torch
from torch.autograd import Variable

from vggs import VGG
from resnets import ResNet20, ResNet56, ResNet110
from densenets import densenet_cifar, DenseNet121
from playground import Conv_Net, Conv_Recusive_Net, Conv_Custom_Recusive_Net

vgg9  = VGG('VGG9')
vgg13 = VGG('VGG13')
vgg19 = VGG('VGG19')
    
resnet20 = ResNet20()
resnet56 = ResNet56()
resnet110 = ResNet110()

densenet_ = densenet_cifar()
densenet121 = DenseNet121()


