import torch
import torch.nn as nn

from functools import partial
from resnet.layer.blocks.blocks import Conv2dAuto, ResNetBasicBlock

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

def resnet_layer_example():
    dummy = torch.ones((1, 32, 48, 48))
    conv3x3 = partial(Conv2dAuto, kernel_size=3, bias=False)
    layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3, conv=conv3x3)
    # layer(dummy).shape
    print(layer)


if __name__ == '__main__':
    resnet_layer_example()