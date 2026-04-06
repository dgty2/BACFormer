import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_la_geometry(mask_2d, pixel_spacing=None):
    """
    计算左心房的几何参数（面积和长轴长度），针对镜像影像进行左侧截断处理
    
    该函数自动处理右侧左心房区域，通过去除左侧的左心室部分，保留右侧62.5%的左心房区域
    使用固定0.3mm像素间距进行计算
    
    Args:
        mask_2d (numpy.ndarray): 二维掩码图像，表示左心房分割结果
        pixel_spacing (float, optional): 像素间距（mm），当前固定为0.3mm
        
    Returns:
        tuple: (area, long_axis_len)
            - area (float): 左心房面积（mm²），保留两位小数
            - long_axis_len (float): 长轴长度（mm），保留两位小数
            
    Note:
        如果掩码为空（所有像素值为0），返回(0.0, 0.0)
    """
    # 检查掩码是否为空，如果全为0则直接返回0
    if np.sum(mask_2d) == 0:
        return 0.0, 0.0
    
    # 固定像素间距为0.3mm，与最初的正常结果对齐
    pixel_spacing = 0.3
    
    # 获取掩码图像的宽度
    w = mask_2d.shape[1]
    # 针对镜像影像进行左侧截断：去掉左侧37.5%，保留右侧62.5%的左心房区域
    mask_right = mask_2d[:, int(w * 0.375):].copy()
    # 将掩码转换为0-255的uint8格式，用于OpenCV处理
    mask_8u = (mask_right * 255).astype(np.uint8)
    # 使用OpenCV查找外部轮廓，采用简单链近似方法
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 初始化长轴长度和面积为0
    long_axis_len = 0.0
    area = 0.0
    
    # 如果找到轮廓，则继续处理
    if len(contours) > 0:
        # 初始化左心房轮廓变量和最大面积变量
        la_contour = None
        max_area = 0
        # 遍历所有轮廓，找到面积最大的轮廓作为左心房
        for cnt in contours:
            # 计算当前轮廓的面积
            area_cnt = cv2.contourArea(cnt)
            # 如果当前轮廓面积大于已知最大面积，则更新
            if area_cnt > max_area:
                max_area = area_cnt
                la_contour = cnt
        
        # 如果找到了左心房轮廓，则计算面积和长轴
        if la_contour is not None:
            # 创建一个与掩码相同大小的空白图像
            temp_mask = np.zeros_like(mask_8u)
            # 在空白图像上绘制左心房轮廓并填充
            cv2.drawContours(temp_mask, [la_contour], -1, 1, thickness=cv2.FILLED)
            # 统计左心房的像素数量
            la_pixels = np.sum(temp_mask)
            # 计算左心房面积：像素数乘以像素间距的平方
            area = la_pixels * (pixel_spacing ** 2)
            
            # 提取轮廓的y坐标（第1列）
            y_coords = la_contour[:, 0, 1]
            # 提取轮廓的x坐标（第0列）
            x_coords = la_contour[:, 0, 0]
            # 找到y坐标最小值的索引（顶部点）
            top_idx = np.argmin(y_coords)
            # 找到y坐标最大值的索引（底部点）
            bottom_idx = np.argmax(y_coords)
            # 获取顶部点的坐标(x, y)
            top_point = (x_coords[top_idx], y_coords[top_idx])
            # 获取底部点的坐标(x, y)
            bottom_point = (x_coords[bottom_idx], y_coords[bottom_idx])
            # 计算顶部点和底部点之间的欧氏距离（像素单位）
            pixel_dist = np.sqrt(((top_point[0] - bottom_point[0]) ** 2) + ((top_point[1] - bottom_point[1]) ** 2))
            # 将像素距离转换为实际毫米长度
            long_axis_len = pixel_dist * pixel_spacing
            
    # 返回四舍五入后的面积和长轴长度（保留两位小数）
    return round(area, 2), round(long_axis_len, 2)


def calculate_lavmax_single_plane(area, long_axis_len):
    """
    使用单平面面积-长度法计算左心房最大容积（LAVmax）
    
    基于椭圆体模型公式：V = (8 × A²) / (3π × L)
    其中A为面积，L为长轴长度
    
    Args:
        area (float): 左心房面积（mm²）
        long_axis_len (float): 左心房长轴长度（mm）
        
    Returns:
        float: 左心房最大容积（ml），保留两位小数
               如果输入参数为0，返回0.0
    """
    # 如果长轴长度或面积为0，无法计算容积，返回0
    if long_axis_len == 0 or area == 0:
        return 0.0
    
    # 使用单平面Simpson公式计算LAVmax（立方毫米）
    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    # 将立方毫米转换为毫升（1ml = 1000mm³）
    lavmax_ml = lavmax_mm3 / 1000
    # 返回四舍五入后的结果（保留两位小数）
    return round(lavmax_ml, 2)


def calculate_dice(pred_mask, gt_mask):
    """
    计算预测掩码与真实掩码之间的Dice系数
    
    针对镜像影像进行左侧截断处理，只比较右侧62.5%区域的相似度
    
    Args:
        pred_mask (numpy.ndarray): 预测的二值掩码图像
        gt_mask (numpy.ndarray): 真实的二值掩码图像
        
    Returns:
        float: Dice系数，范围[0, 1]，保留三位小数
               如果并集为空，返回1.0表示完全匹配
    """
    # 获取预测掩码的宽度
    w = pred_mask.shape[1]
    # 对预测掩码进行左侧截断，保留右侧62.5%区域
    pred_right = pred_mask[:, int(w * 0.375):].copy()
    # 对真实掩码进行相同的左侧截断处理
    gt_right = gt_mask[:, int(w * 0.375):].copy()
    # 将预测掩码转换为二值浮点数（大于0的为1，否则为0）
    pred_bin = (pred_right > 0).astype(np.float32)
    # 将真实掩码转换为二值浮点数
    gt_bin = (gt_right > 0).astype(np.float32)
    # 计算预测和真实的交集（对应位置都为1的像素和）
    intersection = np.sum(pred_bin * gt_bin)
    # 计算预测和真实的并集（两者像素数的总和）
    union = np.sum(pred_bin) + np.sum(gt_bin)
    # 如果并集为0（两个掩码都为空），返回1.0表示完全匹配
    if union == 0:
        return 1.0
    # 计算并返回Dice系数：2×交集/并集，保留三位小数
    return round(2 * intersection / union, 3)


class DiceBoundaryLoss(nn.Module):
    """
    训练用的Dice边界损失函数
    
    结合Dice损失和边界损失，用于医学图像分割任务
    完全还原最初版本的实现
    
    Attributes:
        smooth (float): 平滑项，防止除零错误，默认1e-6
    """
    
    def __init__(self, smooth=1e-6):
        """
        初始化DiceBoundaryLoss
        
        Args:
            smooth (float): 平滑常数，默认为1e-6
        """
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 保存平滑常数，用于防止Dice计算时的除零错误
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        前向传播计算损失
        
        Args:
            pred (torch.Tensor): 预测输出，形状为(B, C, H, W)
            target (torch.Tensor): 真实标签，形状为(B, H, W)或(B, 1, H, W)
            
        Returns:
            torch.Tensor: 标量损失值（1 - mean(Dice)）
        """
        # 获取预测张量的维度：批次大小、类别数、高度、宽度
        B, C, H, W = pred.shape
        
        # 检查目标标签的空间尺寸是否与预测不匹配
        if target.shape[1:] != (H, W):
            # 使用最近邻插值调整目标标签尺寸以匹配预测
            target = F.interpolate(
                target.unsqueeze(1).float(),  # 增加通道维度并转为浮点数
                size=(H, W),  # 目标尺寸
                mode='nearest'  # 使用最近邻插值保持标签整数值
            ).squeeze(1).long()  # 移除通道维度并转回长整型
        
        # 将所有非0的标签值设为1，实现二分类（背景=0，左心房=1）
        target[target > 0] = 1
        # 对预测输出应用softmax函数，得到每个类别的概率分布
        pred = F.softmax(pred, dim=1)
        # 创建一个与预测张量形状相同的全零张量，用于one-hot编码
        target_one_hot = torch.zeros_like(pred)
        # 将目标标签转换为one-hot编码格式
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        # 计算预测概率和one-hot标签的逐元素乘积，并在空间维度求和得到交集
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        # 计算预测概率和one-hot标签的和，并在空间维度求和得到并集
        union = torch.sum(pred + target_one_hot, dim=(2, 3))
        # 计算Dice系数：(2×交集+平滑项)/(并集+平滑项)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        # 返回Dice损失：1减去Dice系数的均值
        return 1 - torch.mean(dice)