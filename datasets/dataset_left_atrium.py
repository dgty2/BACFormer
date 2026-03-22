import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom


class Dataset_LeftAtrium(Dataset):
    def __init__(self, root_path, list_dir, split="train", img_size=224):
        self.root_path = root_path
        self.split = split
        self.img_size = img_size

        list_path = os.path.join(list_dir, f"{split}.txt")
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]

        # 匹配你的路径：img/label 在 lists 文件夹下
        self.img_dir = os.path.join(root_path, "lists", "img")
        self.label_dir = os.path.join(root_path, "lists", "label")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        img_path = os.path.join(self.img_dir, f"{name}.nii.gz")
        label_path = os.path.join(self.label_dir, f"{name}_gt.nii.gz")

        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        # 缩放
        img = zoom(img, (self.img_size / img.shape[0], self.img_size / img.shape[1]), order=1)
        label = zoom(label, (self.img_size / label.shape[0], self.img_size / label.shape[1]), order=0)

        # 单通道张量 [1, H, W]
        img = torch.from_numpy(img).float().unsqueeze(0)
        label = torch.from_numpy(label).long()

        return img, label