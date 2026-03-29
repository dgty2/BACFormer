import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# 临床测量函数，自动过滤多余标注，只计算左心房
def calculate_la_geometry(mask_2d, pixel_spacing=None):
    if np.sum(mask_2d) == 0:
        return 0.0, 0.0
    # 固定0.3mm像素间距，和你最初的正常结果完全对齐
    pixel_spacing = 0.3

    # 调整右侧截断比例：从50%改为62.5%，保留更多左心房
    w = mask_2d.shape[1]
    mask_left = mask_2d[:, :int(w * 0.625)].copy()  # 只截断最右侧的左心室部分

    mask_8u = (mask_left * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_axis_len = 0.0
    area = 0.0
    if len(contours) > 0:
        # 找最大的轮廓，就是左心房
        la_contour = None
        max_area = 0
        for cnt in contours:
            area_cnt = cv2.contourArea(cnt)
            if area_cnt > max_area:
                max_area = area_cnt
                la_contour = cnt

        if la_contour is not None:
            # 只计算左心房的像素
            temp_mask = np.zeros_like(mask_8u)
            cv2.drawContours(temp_mask, [la_contour], -1, 1, thickness=cv2.FILLED)
            la_pixels = np.sum(temp_mask)
            area = la_pixels * (pixel_spacing ** 2)

            # 计算长轴
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
    # 同步调整Dice计算的截断比例
    w = pred_mask.shape[1]
    pred_left = pred_mask[:, :int(w * 0.625)].copy()
    gt_left = gt_mask[:, :int(w * 0.625)].copy()

    pred_bin = (pred_left > 0).astype(np.float32)
    gt_bin = (gt_left > 0).astype(np.float32)
    intersection = np.sum(pred_bin * gt_bin)
    union = np.sum(pred_bin) + np.sum(gt_bin)
    if union == 0:
        return 1.0
    return round(2 * intersection / union, 3)


# 训练用的损失函数，完全还原你最初版本的
class DiceBoundaryLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        # 标签插值，防止尺寸不匹配
        if target.shape[1:] != (H, W):
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=(H, W),
                mode='nearest'
            ).squeeze(1).long()
        # 二分类，所有非0都是左心房
        target[target > 0] = 1
        pred = F.softmax(pred, dim=1)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        union = torch.sum(pred + target_one_hot, dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - torch.mean(dice)