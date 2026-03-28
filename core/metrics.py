import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import find_objects


def calculate_la_geometry(mask_slice, pixdim):
    if np.sum(mask_slice) == 0:
        return 0.0, 0.0
    pixel_area = pixdim[1] * pixdim[2]
    area = np.sum(mask_slice) * pixel_area
    slices = find_objects(mask_slice.astype(int))
    if not slices:
        return float(area), 0.0
    y_slice, x_slice = slices[0]
    h = y_slice.stop - y_slice.start
    w = x_slice.stop - x_slice.start
    long_axis_pix = max(h, w)
    long_axis = long_axis_pix * pixdim[1]
    return float(area), float(long_axis)


def calculate_lavmax_single_plane(area, long_axis):
    if area == 0 or long_axis == 0:
        return 0.0
    volume_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis)
    volume_ml = volume_mm3 / 1000.0
    return float(volume_ml)


def calculate_dice(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.float32)
    gt = (gt_mask > 0).astype(np.float32)
    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt)
    if union == 0:
        return 1.0
    dice = (2. * intersection) / union
    return float(min(dice, 1.0))


# 标准DiceBoundaryLoss，修复了非连续tensor的报错
class DiceBoundaryLoss(nn.Module):
    def __init__(self, sigmoid=False):
        super(DiceBoundaryLoss, self).__init__()
        self.sigmoid = sigmoid

    def forward(self, input, target):
        if self.sigmoid:
            input = F.sigmoid(input)
        else:
            input = F.softmax(input, dim=1)[:, 1, ...]

        input_flat = input.reshape(-1)
        target_flat = target.float().reshape(-1)

        intersection = (input_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + 1e-8) / (input_flat.sum() + target_flat.sum() + 1e-8)

        input_grad = self._get_gradient(input.unsqueeze(1))
        target_grad = self._get_gradient(target.unsqueeze(1).float())
        grad_loss = F.l1_loss(input_grad, target_grad)

        return dice_loss + grad_loss

    def _get_gradient(self, x):
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        grad_x = F.pad(grad_x, (1, 0, 0, 0), mode='constant', value=0)
        grad_y = F.pad(grad_y, (0, 0, 1, 0), mode='constant', value=0)
        return torch.cat([grad_x, grad_y], dim=1)