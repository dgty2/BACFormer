#!/usr/bin/env python
# BACFormer 左心房分割专用工具/损失函数
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- 损失函数 ----------------------
class DiceLoss(nn.Module):
    """Dice损失，解决医学图像类别不平衡问题"""
    def __init__(self, smooth=1e-8):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class BoundaryLoss(nn.Module):
    """边界损失，强化左心房轮廓精度"""
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        # 3x3平均池化提取边界
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9
        pred_pool = F.conv2d(pred, kernel, padding=1)
        target_pool = F.conv2d(target.float(), kernel, padding=1)
        pred_edge = torch.abs(pred - pred_pool)
        target_edge = torch.abs(target.float() - target_pool)
        return F.mse_loss(pred_edge, target_edge)

class DiceBoundaryLoss(nn.Module):
    """组合损失：70% DiceLoss + 30% BoundaryLoss"""
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        return 0.7 * self.dice_loss(pred, target) + 0.3 * self.boundary_loss(pred, target)

# ---------------------- 评估指标 ----------------------
def calculate_dice(pred, target):
    """计算二分类Dice系数（左心房分割金标准）"""
    pred = torch.sigmoid(pred) > 0.5  # 二值化
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.item()