import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


class DiceBoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        # 1. 先把标签插值到和模型输出同尺寸（防止尺寸不匹配）
        if target.shape[1:] != (H, W):
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()

        # 2. ✅ 核心修复：二分类任务，所有非0像素都视为类别1（彻底解决越界）
        target[target > 0] = 1

        pred = F.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)  # [B, 2, H, W]
        # 3. scatter_ 维度完全匹配（不会再越界）
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred + target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice)


# 临床测量函数（保持不变，已修复拼写）
def calculate_la_geometry(mask_2d, pixel_spacing):
    la_pixels = np.sum(mask_2d == 1)
    area = la_pixels * (pixel_spacing ** 2)

    mask_8u = (mask_2d * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_axis_len = 0.0
    if len(contours) > 0:
        la_contour = max(contours, key=cv2.contourArea)
        y_coords = la_contour[:, 0, 1]
        x_coords = la_contour[:, 0, 0]

        top_idx = np.argmin(y_coords)
        bottom_idx = np.argmax(y_coords)
        top_point = (x_coords[top_idx], y_coords[top_idx])
        bottom_point = (x_coords[bottom_idx], y_coords[bottom_idx])

        pixel_dist = np.sqrt(((top_point[0] - bottom_point[0]) ** 2) + ((top_point[1] - bottom_point[1]) ** 2))
        long_axis_len = pixel_dist * pixel_spacing

    return round(area, 2), round(long_axis_len, 2)


def calculate_lavmax_single_plane(area, long_axis_len):
    if long_axis_len == 0 or area == 0:
        return 0.0
    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    lavmax_ml = lavmax_mm3 / 1000
    return round(lavmax_ml, 2)


def calculate_dice(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)

    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    if union == 0:
        return 1.0
    return round(2 * intersection / union, 3)