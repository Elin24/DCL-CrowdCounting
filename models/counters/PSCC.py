from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *
import os

model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class PSCC(nn.Module):
    def __init__(self, pretrained=True):
        super(PSCC, self).__init__()
        vgg = models.vgg16()
        if pretrained:
            vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.encoder = nn.Sequential(*features[0:23])


        self.decoder = nn.Sequential(Conv2d(512, 256, 3, same_padding=True, NL='relu'), # (256, h, w)
                                     Conv2d(256,  64, 3, same_padding=True, NL='relu'),
                                     nn.PixelShuffle(8),
                                     nn.ReLU()) # (64, h*2, w*2)

        # weights_normal_init(self.decoder)    
        

    def forward(self, x):
        x = self.encoder(x)       
        x = self.decoder(x)

        return x