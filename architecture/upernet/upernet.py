import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvModule
from .resnet import Resnet
from .uper_head import UperHead


class UperNet(nn.Module):
    def __init__(self):
        super(UperNet, self).__init__()
        self.backbone = Resnet()
        self.head = UperHead([256, 512, 1024, 2048])
        self.conv1 = ConvModule(9, 6, 3, p=1)
        
    def forward(self, x):
        features = self.backbone(x)
        outs = self.head(features)
        outs = F.interpolate(outs, x.shape[-2:], mode='bilinear')
        outs = torch.cat([outs, x], dim = 1)
        outs = self.conv1(outs)
        return outs