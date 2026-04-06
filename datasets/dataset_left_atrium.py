import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom


class Dataset_LeftAtrium(Dataset):
    """
    左心房数据集类：加载和管理医学影像数据
    
    从文本列表文件读取样本名称，加载对应的影像和标签文件
    自动进行数据预处理（二值化、缩放）
    """
    
    def __init__(self, root_path, list_dir, split="train", img_size=224):
        """
        初始化数据集
        
        Args:
            root_path (str): 项目根路径
            list_dir (str): 列表文件所在目录
            split (str): 数据集划分（"train" 或 "test"）
            img_size (int): 输出图像尺寸
        """
        self.root_path = root_path
        self.split = split
        self.img_size = img_size
        list_path = os.path.join(list_dir, f"{split}.txt")
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]
        
        self.img_dir = os.path.join(root_path, "lists", "img")
        self.label_dir = os.path.join(root_path, "lists", "label")

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 样本数量
        """
        return len(self.names)

    def __getitem__(self, idx):
        """
        获取单个样本数据
        
        加载影像和标签，进行二值化处理和尺寸缩放
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (img, label)
                - img: 图像张量 [1, H, W]
                - label: 标签张量 [H, W]
        """
        name = self.names[idx]
        img_path = os.path.join(self.img_dir, f"{name}.nii.gz")
        label_path = os.path.join(self.label_dir, f"{name}_gt.nii.gz")
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()

        label = (label > 0).astype(np.float32)

        img = zoom(img, (self.img_size / img.shape[0], self.img_size / img.shape[1]), order=1)
        label = zoom(label, (self.img_size / label.shape[0], self.img_size / label.shape[1]), order=0)

        img = torch.from_numpy(img).float().unsqueeze(0)
        label = torch.from_numpy(label).long()
        return img, label