#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice损失函数：解决医学图像类别不平衡问题
    
    通过计算预测和真实标签之间的Dice系数来衡量相似度
    损失值为 1 - Dice系数
    """
    
    def __init__(self, smooth=1e-8):
        """
        初始化DiceLoss
        
        Args:
            smooth (float): 平滑项，防止除零错误，默认1e-8
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        计算Dice损失
        
        Args:
            pred (torch.Tensor): 预测输出（未归一化的logits）
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: Dice损失值
        """
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    边界损失函数：强化左心房轮廓精度
    
    通过提取预测和真实标签的边缘信息，计算边缘的MSE损失
    有助于提高分割边界的准确性
    """
    
    def forward(self, pred, target):
        """
        计算边界损失
        
        使用3x3平均池化核提取边缘特征，然后计算边缘差异的MSE
        
        Args:
            pred (torch.Tensor): 预测输出
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 边界损失值
        """
        pred = torch.sigmoid(pred)
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9
        pred_pool = F.conv2d(pred, kernel, padding=1)
        target_pool = F.conv2d(target.float(), kernel, padding=1)
        pred_edge = torch.abs(pred - pred_pool)
        target_edge = torch.abs(target.float() - target_pool)
        return F.mse_loss(pred_edge, target_edge)


class DiceBoundaryLoss(nn.Module):
    """
    组合损失函数：70% DiceLoss + 30% BoundaryLoss
    
    结合区域相似度和边界精度的优势，提供更全面的监督信号
    """
    
    def __init__(self):
        """初始化组合损失函数"""
        super().__init__()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, pred, target):
        """
        计算组合损失
        
        Args:
            pred (torch.Tensor): 预测输出
            target (torch.Tensor): 真实标签
            
        Returns:
            torch.Tensor: 加权组合损失值
        """
        return 0.7 * self.dice_loss(pred, target) + 0.3 * self.boundary_loss(pred, target)


def calculate_dice(pred, target):
    """
    计算二分类Dice系数（左心房分割金标准）
    
    Args:
        pred (torch.Tensor): 预测输出（未归一化的logits）
        target (torch.Tensor): 真实标签
        
    Returns:
        float: Dice系数值
    """
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    return dice.item()