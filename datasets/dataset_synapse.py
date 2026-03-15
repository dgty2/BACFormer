#!/usr/bin/env python
# 左心房分割数据集加载（适配项目lists文件）
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

class SynapseDataset(Dataset):
    def __init__(self, root_path, list_dir, split="train", img_size=224):
        """
        Args:
            root_path: 数据集根目录（包含img/和label/文件夹）
            list_dir: 列表文件目录（lists/lists_Synapse/）
            split: train/val/test
            img_size: 输入图像尺寸
        """
        self.root_path = root_path
        self.split = split
        self.img_size = img_size

        # 读取样本列表（原项目用txt文件管理样本）
        list_file = os.path.join(list_dir, f"{split}.txt")
        with open(list_file, "r") as f:
            self.samples = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]

        # 加载影像和标签（nii.gz格式）
        img_path = os.path.join(self.root_path, "img", f"{sample_id}.nii.gz")
        label_path = os.path.join(self.root_path, "label", f"{sample_id}.nii.gz")

        img_nii = nib.load(img_path)
        img = img_nii.get_fdata()
        label = nib.load(label_path).get_fdata()

        # 论文要求预处理：裁剪+归一化
        img = np.clip(img, -125, 275)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # 取中间层（3D→2D切片，适配训练输入）
        slice_idx = img.shape[-1] // 2
        img = img[:, :, slice_idx]
        label = label[:, :, slice_idx]

        # 转为张量并适配模型输入
        img = torch.from_numpy(img).float().unsqueeze(0).repeat(3, 1, 1)  # 单通道→3通道
        label = torch.from_numpy(label).float().unsqueeze(0)  # [1, H, W]

        return {"image": img, "label": label}