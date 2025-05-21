from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .wide_block import WideBlock


settings = {
    '50': [[3, 4, 6, 3], [256, 512, 1024, 2048]],
    '101': [[3, 4, 23, 3], [256, 512, 1024, 2048]],}
class Resnet(nn.Module):
    def __init__(self, setting:str = '50'):
        super(Resnet, self).__init__()
        assert setting in settings.keys(), f"ResNet model name should be in {list(settings.keys())}"
        depths, channels = settings[setting]

        self.inplanes = 64
        self.channels = channels
        self.conv1 = nn.Conv2d(3, self.inplanes, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, depths[0], s=1)
        self.layer2 = self._make_layer(128, depths[1], s=2)
        self.layer3 = self._make_layer(256, depths[2], s=2)
        self.layer4 = self._make_layer(512, depths[3], s=2)
        
    def _make_layer(self, planes, depth, s=1) -> nn.Sequential:
        downsample = None
        if s != 1 or self.inplanes != planes * WideBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * WideBlock.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * WideBlock.expansion)
            )
        layers = nn.Sequential(
            WideBlock(self.inplanes, planes, s, downsample),
            *[WideBlock(planes * WideBlock.expansion, planes) for _ in range(1, depth)]
        )
        self.inplanes = planes * WideBlock.expansion
        return layers


    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))   # [64, H/4, W/4]
        x1 = self.layer1(x)  # [64/256, H/4, W/4]   
        x2 = self.layer2(x1)  # [128/512, H/8, W/8]
        x3 = self.layer3(x2)  # [256/1024, H/16, W/16]
        x4 = self.layer4(x3)  # [512/2048, H/32, W/32]
        return x1, x2, x3, x4