import torch
import torch.nn as nn

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict


class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding =  (self.kernel_size[0] // 2, self.kernel_size[1] // 2) # dynamic add padding based on the kernel_size
        

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels =  in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=0, conv=None, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = nn.Sequential(OrderedDict(
        {
            'conv' : nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1,
                      stride=self.downsampling, bias=False),
            'bn' : nn.BatchNorm2d(self.expanded_channels)
            
        })) if self.should_apply_shortcut else None
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            self.conv_bn(self.in_channels, self.out_channels, self.conv, bias=False, stride=self.downsampling),
            activation(),
            self.conv_bn(self.out_channels, self.expanded_channels, self.conv, bias=False),
        )

    def conv_bn(self, in_channels, out_channels, conv, *args, **kwargs):
        return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))



class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           self.conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activation(),
             self.conv_bn(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activation(),
             self.conv_bn(self.out_channels, self.expanded_channels, self.conv, kernel_size=1),
        )

    def conv_bn(self, in_channels, out_channels, conv, *args, **kwargs):
        return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))


def residual_block_ex():
    dummy = torch.ones((1, 1, 1, 1))
    block = ResidualBlock(1, 64)
    print(block(dummy))

def resnet_example():
    dummy = torch.ones((1, 32, 224, 224))
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    block = ResNetBasicBlock(32, 64, conv=conv3x3)
    block(dummy).shape
    print(block)

def resnet_bottleneck():
    dummy = torch.ones((1, 32, 10, 10))
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    block = ResNetBottleNeckBlock(32, 64, conv=conv3x3)
    block(dummy).shape
    print(block)


def run_conv_ex():
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    conv = conv3x3(in_channels=32, out_channels=64)
    print(conv)
    del conv

def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(OrderedDict({'conv': conv(in_channels, out_channels, *args, **kwargs), 
                          'bn': nn.BatchNorm2d(out_channels) }))



if __name__ == '__main__':
    # run_conv_ex()
    # residual_block_ex()
    # print(ResNetResidualBlock(32, 64))
    # print(conv_bn(3, 3, nn.Conv2d, kernel_size=3))
    # resnet_example()
    # resnet_bottleneck()
    # resnet_layer_example()