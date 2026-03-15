#!/usr/bin/env python
# BACFormer 左心房分割专用模型
import torch
import torch.nn as nn

class Encoder(nn.Module):
    """编码器：提取多尺度特征"""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU())

    def forward(self, x):
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]

class Decoder(nn.Module):
    """解码器：还原特征图尺寸"""
    def __init__(self):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)

    def forward(self, features):
        c1, c2, c3, c4 = features
        d4 = self.up4(c4) + c3
        d3 = self.up3(d4) + c2
        d2 = self.up2(d3) + c1
        return d2

class BACFormer(nn.Module):
    """BACFormer主模型（左心房二分类专用）"""
    def __init__(self, num_classes=1, img_size=224):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)  # 输出1类（左心房/背景）

    def forward(self, x):
        features = self.encoder(x)
        decoder_out = self.decoder(features)
        out = self.out_conv(decoder_out)
        return out