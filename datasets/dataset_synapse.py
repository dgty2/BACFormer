import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib


# ========== 新增：自动定位项目根目录（解决路径问题） ==========
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "lists")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


PROJECT_ROOT = get_project_root()


# ============================================================

class Dataset_Synapse(Dataset):
    def __init__(self, root_path, list_dir, split, img_size=256, norm_x=True):
        # 拼接绝对路径
        self.root_path = os.path.join(PROJECT_ROOT, root_path)
        self.list_dir = os.path.join(PROJECT_ROOT, list_dir)
        self.split = split
        self.img_size = img_size
        self.norm_x = norm_x

        self.list_file = os.path.join(self.list_dir, f'{split}.txt')
        self.samples = [line.strip() for line in open(self.list_file, encoding='utf-8').readlines()]
        print(f"📌 加载{split}集，共{len(self.samples)}个样本（1-100号）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        # 1. 读取影像文件
        img_path = os.path.join(self.root_path, "img", f"{sample_id}.nii.gz")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ 影像文件不存在：{img_path}")

        img_nii = nib.load(img_path)
        img_data = img_nii.get_fdata()  # 读取数据
        pixdim = img_nii.header['pixdim']  # 像素间距
        pixel_spacing = (pixdim[1] + pixdim[2]) / 2  # mm/像素

        # ========== 核心修复：兼容2D/3D影像 ==========
        if img_data.ndim == 2:
            # 影像本身是2D（H,W），无需切片
            img_2d = img_data.astype(np.float32)
            es_slice_idx = 0  # 2D无深度维度，标记为0
        elif img_data.ndim == 3:
            # 影像是3D（H,W,D），取中间帧
            es_slice_idx = img_data.shape[-1] // 2
            img_2d = img_data[:, :, es_slice_idx].astype(np.float32)
        else:
            raise ValueError(f"❌ 不支持的影像维度：{img_data.ndim}（仅支持2D/3D）")
        # =============================================

        # 2. 预处理：归一化 + 维度调整 (1, H, W)
        if self.norm_x:
            img_2d = (img_2d - img_2d.min()) / (img_2d.max() - img_2d.min() + 1e-8)
        img_2d = np.expand_dims(img_2d, axis=0)  # 增加通道维度

        # 3. 读取标签文件（兼容2D/3D）
        label_path = os.path.join(self.root_path, "label", f"{sample_id}_gt.nii.gz")
        label_2d = np.zeros_like(img_2d)  # 无标签时默认0
        if os.path.exists(label_path):
            label_nii = nib.load(label_path)
            label_data = label_nii.get_fdata()
            # 标签也适配2D/3D
            if label_data.ndim == 2:
                label_2d = np.expand_dims(label_data, axis=0).astype(np.int64)
            elif label_data.ndim == 3:
                label_2d = np.expand_dims(label_data[:, :, es_slice_idx], axis=0).astype(np.int64)

        return {
            'image': torch.from_numpy(img_2d).float(),
            'label': torch.from_numpy(label_2d).long(),
            'sample_id': sample_id,
            'slice_idx': es_slice_idx,
            'pixel_spacing': pixel_spacing
        }


# 测试数据加载（无需改路径，自动适配）
if __name__ == "__main__":
    dataset = Dataset_Synapse(
        root_path="lists",
        list_dir="lists/lists_Synapse",
        split="test"
    )
    # 测试第一个样本
    sample = dataset[0]
    print(f"✅ 样本ID: {sample['sample_id']}")
    print(f"影像形状: {sample['image'].shape}")
    print(f"像素间距: {sample['pixel_spacing']} mm/像素")
    print(f"影像维度: {sample['image'].ndim}（正常应为3：[C,H,W]）")