import torch.nn as nn

from .conv import ConvBNReLU
from .linear_attn import LinearAttention
from .rest import rest_lite
from .upsample import UpSample

class Attention_Embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention_Embedding, self).__init__()
        self.attention = LinearAttention(in_channels)
        self.conv_attn = ConvBNReLU(in_channels, out_channels)
        self.up = UpSample(out_channels)

    def forward(self, high_feat, low_feat):
        A = self.attention(high_feat)
        A = self.conv_attn(A)
        A = self.up(A)

        output = low_feat * A
        output += low_feat

        return output


class DependencyPath(nn.Module):
    def __init__(self, weight_path='pretrain_weights/rest_lite.pth'):
        super(DependencyPath, self).__init__()
        self.ResT = rest_lite(weight_path=weight_path)
        self.AE = Attention_Embedding(512, 256)
        self.conv_avg = ConvBNReLU(256, 128, ks=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2.)

    def forward(self, x):
        e3, e4 = self.ResT(x)

        e = self.conv_avg(self.AE(e4, e3))

        return self.up(e)