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
        # 保存项目根路径到实例变量
        self.root_path = root_path
        # 保存数据集划分类型（训练集或测试集）
        self.split = split
        # 保存目标图像尺寸
        self.img_size = img_size
        # 构建列表文件的完整路径，根据split参数选择train.txt或test.txt
        list_path = os.path.join(list_dir, f"{split}.txt")
        # 打开列表文件并读取所有行，去除每行的空白字符后存储为样本名称列表
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines()]
        
        # 构建影像文件目录路径：root_path/lists/img
        self.img_dir = os.path.join(root_path, "lists", "img")
        # 构建标签文件目录路径：root_path/lists/label
        self.label_dir = os.path.join(root_path, "lists", "label")

    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 样本数量
        """
        # 返回样本名称列表的长度，即数据集中的样本总数
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
        # 根据索引获取当前样本的名称
        name = self.names[idx]
        # 构建影像文件的完整路径
        img_path = os.path.join(self.img_dir, f"{name}.nii.gz")
        # 构建标签文件的完整路径（GT文件命名格式：原名_gt.nii.gz）
        label_path = os.path.join(self.label_dir, f"{name}_gt.nii.gz")
        # 使用nibabel加载影像文件并提取数值数据
        img = nib.load(img_path).get_fdata()
        # 使用nibabel加载标签文件并提取数值数据
        label = nib.load(label_path).get_fdata()

        # 将标签二值化：所有大于0的像素设为1.0（左心房），0保持为0（背景）
        # 这解决了原始标签可能包含多类别或非整数值的问题
        label = (label > 0).astype(np.float32)

        # 使用双线性插值（order=1）缩放图像到目标尺寸，保持图像平滑
        # 分别计算高度和宽度的缩放比例
        img = zoom(img, (self.img_size / img.shape[0], self.img_size / img.shape[1]), order=1)
        # 使用最近邻插值（order=0）缩放标签，保持标签值不变（避免插值产生非整数值）
        label = zoom(label, (self.img_size / label.shape[0], self.img_size / label.shape[1]), order=0)

        # 将numpy数组转换为PyTorch张量，转为浮点型，并在第0维添加通道维度（H,W -> 1,H,W）
        img = torch.from_numpy(img).float().unsqueeze(0)
        # 将标签转换为PyTorch长整型张量，用于交叉熵损失函数的输入要求
        label = torch.from_numpy(label).long()
        # 返回处理后的图像张量和标签张量元组
        return img, label