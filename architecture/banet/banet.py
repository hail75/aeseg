import torch.nn as nn

from .dependency_path import DependencyPath
from .feature_agg import FeatureAggregationModule
from .output import Output
from .texture_path import TexturePath


class BANet(nn.Module):
    def __init__(self, num_classes=6, weight_path=None):
        super(BANet, self).__init__()
        self.name = 'BANet'

        self.cp = DependencyPath(weight_path=weight_path)
        self.sp = TexturePath()
        self.fam = FeatureAggregationModule(256, 256)
        self.conv_out = Output(256, 256, num_classes, up_factor=8)
        self.init_weight()

    def forward(self, x):
        feat = self.cp(x)
        feat_sp = self.sp(x)
        feat_fuse = self.fam(feat_sp, feat)

        feat_out = self.conv_out(feat_fuse)

        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)