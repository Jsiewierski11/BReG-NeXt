import torch
import torch.nn as nn

from resnet.encoder.encoder import ResNetEncoder
from resnet.decoder.decoder import ResnetDecoder
from resnet.layer.blocks.blocks import ResNetResidualBlock

class BReG_NeXt():
    
    def __init__(self):
        conv32x3 = partial(Conv2dAuto, kernel_size=32 ,3, bias=False)
        net = ResNetResidualBlock(7, 32)
        net = ResNetResidualBlock(1, 64, downsampling=1)
        net = ResNetResidualBlock(8, 64)
        net = ResNetResidualBlock(1, 128, downsampling=1)
        net = ResNetResidualBlock(1, 128)

"""
Tensorflow implementation of the BReG_NeXt net

def BReG_NeXt(_X):
  #BReG_NeXt implementation. Returns feature map before softmax.
  
  net = tflearn.conv_2d(_X, 32, 3, regularizer='L2', weight_decay=0.0001)
  net = residual_block(net, 7, 32,activation='elu')
  net = residual_block(net, 1, 64, downsample=True,activation='elu')
  net = residual_block(net, 8, 64,activation='elu')
  net = residual_block(net, 1, 128, downsample=True,activation='elu')
  net = residual_block(net, 7, 128,activation='elu')
  net = tflearn.batch_normalization(net)
  net = tflearn.activation(net, 'elu')
  net = tflearn.global_avg_pool(net)
  # Regression
  logits = tflearn.fully_connected(net, n_classes, activation='linear')
  return logits
"""

if __name__ == '__main__':