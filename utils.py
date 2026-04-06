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
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 保存平滑常数，用于防止Dice计算时的除零错误
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
        # 对预测输出应用sigmoid函数，将logits转换为0-1之间的概率
        pred = torch.sigmoid(pred)
        # 将预测张量展平为一维向量
        pred = pred.view(-1)
        # 将目标张量展平并转换为浮点型
        target = target.view(-1).float()
        # 计算交集：预测和目标对应位置相乘后求和
        intersection = (pred * target).sum()
        # 计算Dice系数：(2×交集+平滑项)/(预测总和+目标总和+平滑项)
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        # 返回Dice损失：1减去Dice系数
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
        # 对预测输出应用sigmoid函数
        pred = torch.sigmoid(pred)
        # 创建3x3的平均池化核（所有元素为1/9），移动到预测张量的设备上
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9
        # 对预测进行平均池化，padding=1保持尺寸不变
        pred_pool = F.conv2d(pred, kernel, padding=1)
        # 对目标进行平均池化
        target_pool = F.conv2d(target.float(), kernel, padding=1)
        # 提取预测的边缘：原始预测减去池化后的预测
        pred_edge = torch.abs(pred - pred_pool)
        # 提取目标的边缘
        target_edge = torch.abs(target.float() - target_pool)
        # 计算边缘的均方误差损失
        return F.mse_loss(pred_edge, target_edge)


class DiceBoundaryLoss(nn.Module):
    """
    组合损失函数：70% DiceLoss + 30% BoundaryLoss
    
    结合区域相似度和边界精度的优势，提供更全面的监督信号
    """
    
    def __init__(self):
        """初始化组合损失函数"""
        # 调用父类初始化
        super().__init__()
        # 创建Dice损失实例
        self.dice_loss = DiceLoss()
        # 创建边界损失实例
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
        # 按权重组合两个损失：70% Dice + 30% Boundary
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
    # 对预测应用sigmoid并阈值化为0.5，得到二值预测
    pred = torch.sigmoid(pred) > 0.5
    # 将预测展平为一维连续张量
    pred = pred.contiguous().view(-1)
    # 将目标展平并转换为浮点型
    target = target.contiguous().view(-1).float()
    # 计算交集：预测和目标对应位置相乘后求和
    intersection = (pred * target).sum()
    # 计算并集：预测总和加目标总和
    union = pred.sum() + target.sum()
    # 计算Dice系数，添加小值1e-8防止除零
    dice = (2. * intersection + 1e-8) / (union + 1e-8)
    # 返回Python标量值
    return dice.item()