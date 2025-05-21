import timm 
import torch.nn as nn

from .cam import CAM_Module
from .decoder import DecoderBlock
from .nonlinearity import nonlinearity
from .pam import PAM_Module



class PAM_CAM_Layer(nn.Module):
    def __init__(self, in_ch):
        super(PAM_CAM_Layer, self).__init__()
        self.PAM = PAM_Module(in_ch)
        self.CAM = CAM_Module()

    def forward(self, x):
        return self.PAM(x) + self.CAM(x)



class MANet(nn.Module):
    def __init__(self, num_channels=3, num_classes=6, backbone_name='resnet50', pretrained=True):
        super(MANet, self).__init__()
        self.name = 'MANet'

        self.backbone = timm.create_model(backbone_name, features_only=True, output_stride=32,
                                          out_indices=(1, 2, 3, 4), pretrained=pretrained)
        filters = self.backbone.feature_info.channels()
        
        self.attention4 = PAM_CAM_Layer(filters[3])
        self.attention3 = PAM_CAM_Layer(filters[2])
        self.attention2 = PAM_CAM_Layer(filters[1])
        self.attention1 = PAM_CAM_Layer(filters[0])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
                

    def forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.backbone(x)

        e4 = self.attention4(e4)

        # Decoder
        d4 = self.decoder4(e4) + self.attention3(e3)
        d3 = self.decoder3(d4) + self.attention2(e2)
        d2 = self.decoder2(d3) + self.attention1(e1)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out