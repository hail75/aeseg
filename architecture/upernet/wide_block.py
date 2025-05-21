import torch.nn as nn
import torch.nn.functional as F


class WideBlock(nn.Module):
    expansion:int = 4
    def __init__(self, c1, c2, stride = 1, downsample = None):
        super(WideBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        return out