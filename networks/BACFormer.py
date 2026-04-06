import torch
import torch.nn as nn
import timm
import torch.nn.functional as F


class BACFormer(nn.Module):
    """
    BACFormer网络架构：基于ResNet编码器和自定义解码器的医学图像分割模型
    
    该网络采用编码器-解码器结构，带有跳跃连接，专门用于左心房分割任务
    输入为单通道图像，输出为多类别分割结果
    """
    
    def __init__(self, num_classes=2):
        """
        初始化BACFormer模型
        
        Args:
            num_classes (int): 分割类别数，默认为2（背景+左心房）
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()
        
        # 使用timm库创建ResNet18编码器，配置为单通道输入（适配医学影像）
        self.encoder = timm.create_model(
            'resnet18',  # 使用ResNet18作为骨干网络
            in_chans=1,  # 设置输入通道数为1（灰度医学影像）
            pretrained=False,  # 不使用预训练权重，从头开始训练
            features_only=True  # 只返回特征图，不返回分类头输出
        )
        # 获取编码器各层的输出通道数，用于构建解码器 [64, 64, 128, 256, 512]
        enc_channels = self.encoder.feature_info.channels()

        # 构建5层解码器模块，每层逐步恢复空间分辨率并融合编码器特征
        # 第5层解码器：将512通道降维到256通道
        self.decoder5 = self._decoder_block(enc_channels[4], enc_channels[3])
        # 第4层解码器：将256通道降维到128通道
        self.decoder4 = self._decoder_block(enc_channels[3], enc_channels[2])
        # 第3层解码器：将128通道降维到64通道
        self.decoder3 = self._decoder_block(enc_channels[2], enc_channels[1])
        # 第2层解码器：保持64通道不变
        self.decoder2 = self._decoder_block(enc_channels[1], enc_channels[0])
        # 第1层解码器：保持64通道不变，最后输出前的一层
        self.decoder1 = self._decoder_block(enc_channels[0], enc_channels[0])

        # 输出卷积层：将64通道映射到num_classes通道的分割结果
        self.out_conv = nn.Conv2d(enc_channels[0], num_classes, kernel_size=1)

    def _decoder_block(self, in_channels, out_channels):
        """
        构建解码器块，包含转置卷积、批归一化和ReLU激活
        
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            
        Returns:
            nn.Sequential: 解码器块序列
        """
        return nn.Sequential(
            # 使用转置卷积进行2倍上采样，步长为2，核大小为2
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            # 批归一化层，加速训练并提高稳定性
            nn.BatchNorm2d(out_channels),
            # ReLU激活函数，inplace=True节省内存
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        前向传播过程
        
        通过编码器提取多层特征，然后通过解码器逐步恢复空间分辨率
        在每层解码后使用跳跃连接融合编码器特征
        
        Args:
            x (torch.Tensor): 输入张量，形状为[B, 1, 224, 224]
            
        Returns:
            torch.Tensor: 输出张量，形状为[B, num_classes, 224, 224]
        """
        # 通过编码器提取5层特征，尺寸分别为112/56/28/14/7（相对于224输入）
        f1, f2, f3, f4, f5 = self.encoder(x)

        # 第5层解码：将f5（512通道）上采样并与f4（256通道）融合
        d5 = self.decoder5(f5)
        # 使用双线性插值确保d5和f4的空间尺寸完全一致
        d5 = F.interpolate(d5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接：将上采样后的特征与编码器特征相加
        d5 = d5 + f4

        # 第4层解码：将d5（256通道）上采样并与f3（128通道）融合
        d4 = self.decoder4(d5)
        # 插值对齐d4和f3的尺寸
        d4 = F.interpolate(d4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接相加
        d4 = d4 + f3

        # 第3层解码：将d4（128通道）上采样并与f2（64通道）融合
        d3 = self.decoder3(d4)
        # 插值对齐d3和f2的尺寸
        d3 = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接相加
        d3 = d3 + f2

        # 第2层解码：将d3（64通道）上采样并与f1（64通道）融合
        d2 = self.decoder2(d3)
        # 插值对齐d2和f1的尺寸
        d2 = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        # 跳跃连接相加
        d2 = d2 + f1

        # 第1层解码：将d2上采样到原始输入尺寸
        d1 = self.decoder1(d2)
        # 最后一次插值，确保输出尺寸与输入x完全一致（224×224）
        d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)

        # 通过1x1卷积将64通道映射到num_classes通道的分割输出
        out = self.out_conv(d1)
        # 返回分割结果，形状为[B, num_classes, H, W]
        return out