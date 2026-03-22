import torch
import torch.nn as nn
import timm

class BACFormer(nn.Module):
    def __init__(self, num_classes=2):
        super(BACFormer, self).__init__()
        # 单通道输入（适配医学影像）
        self.encoder = timm.create_model(
            'resnet18',
            in_chans=1,
            pretrained=False,
            features_only=True
        )
        enc_ch = self.encoder.feature_info.channels()  # [64, 64, 128, 256, 512]

        # 解码器：逐层上采样，最终输出 224×224
        self.dec5 = nn.ConvTranspose2d(enc_ch[4], enc_ch[3], kernel_size=2, stride=2)  # 7→14
        self.dec4 = nn.ConvTranspose2d(enc_ch[3], enc_ch[2], kernel_size=2, stride=2)  #14→28
        self.dec3 = nn.ConvTranspose2d(enc_ch[2], enc_ch[1], kernel_size=2, stride=2)  #28→56
        self.dec2 = nn.ConvTranspose2d(enc_ch[1], enc_ch[0], kernel_size=2, stride=2)  #56→112
        self.dec1 = nn.ConvTranspose2d(enc_ch[0], enc_ch[0], kernel_size=2, stride=2)  #112→224

        self.out_conv = nn.Conv2d(enc_ch[0], num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, 224, 224]
        f1, f2, f3, f4, f5 = self.encoder(x)  # 各层特征尺寸: 112/56/28/14/7

        d5 = self.dec5(f5) + f4  # 7→14
        d4 = self.dec4(d5) + f3  #14→28
        d3 = self.dec3(d4) + f2  #28→56
        d2 = self.dec2(d3) + f1  #56→112
        d1 = self.dec1(d2)        #112→224

        out = self.out_conv(d1)  # [B, 2, 224, 224] ✅ 与标签尺寸完全匹配
        return out