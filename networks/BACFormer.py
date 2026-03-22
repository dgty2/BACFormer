import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class BACFormer(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 单通道输入（适配医学影像）
        self.encoder = timm.create_model(
            'resnet18',
            in_chans=1,
            pretrained=False,
            features_only=True
        )
        enc_channels = self.encoder.feature_info.channels()  # [64, 64, 128, 256, 512]

        # 解码器模块
        self.decoder5 = self._decoder_block(enc_channels[4], enc_channels[3])  # 512→256
        self.decoder4 = self._decoder_block(enc_channels[3], enc_channels[2])  # 256→128
        self.decoder3 = self._decoder_block(enc_channels[2], enc_channels[1])  # 128→64
        self.decoder2 = self._decoder_block(enc_channels[1], enc_channels[0])  # 64→64
        self.decoder1 = self._decoder_block(enc_channels[0], enc_channels[0])  # 64→64

        self.out_conv = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, 1, 224, 224]
        f1, f2, f3, f4, f5 = self.encoder(x)  # 各层特征尺寸: 112/56/28/14/7

        # ✅ 跳跃连接前先插值对齐尺寸
        d5 = self.decoder5(f5)
        d5 = F.interpolate(d5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        d5 = d5 + f4  # 尺寸对齐后相加

        d4 = self.decoder4(d5)
        d4 = F.interpolate(d4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + f3

        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + f2

        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + f1

        d1 = self.decoder1(d2)  # 最后上采样到 224×224
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)

        out = self.out_conv(d1)  # [B, 2, 224, 224]
        return out