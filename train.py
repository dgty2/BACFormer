#!/usr/bin/env python
# BACFormer 左心房分割训练启动脚本
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.dataset_synapse import Dataset_Synapse
from networks.BACFormer import BACFormer
from trainer import train
import os

def main():
    parser = argparse.ArgumentParser(description="BACFormer 左心房分割训练")
    parser.add_argument("--root_path", default="./data/Synapse", type=str, help="数据集根目录（含img/和label/）")
    parser.add_argument("--list_dir", default="./lists/lists_Synapse", type=str, help="样本列表目录")
    parser.add_argument("--num_classes", default=1, type=int, help="左心房=1 类（二分类）")
    parser.add_argument("--img_size", default=224, type=int, help="输入图像尺寸")
    parser.add_argument("--batch_size", default=1, type=int, help="批次大小（小显存电脑用 1）")
    parser.add_argument("--max_epochs", default=50, type=int, help="训练轮数")
    parser.add_argument("--base_lr", default=0.01, type=float, help="初始学习率")
    parser.add_argument("--save_path", default="./model_out", type=str, help="模型保存路径")
    args = parser.parse_args()

    # 创建模型保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 加载数据集
    train_dataset = Dataset_Synapse(args.root_path, args.list_dir, split="train", img_size=args.img_size)
    val_dataset = Dataset_Synapse(args.root_path, args.list_dir, split="test_vol", img_size=args.img_size)  # 原项目用 test_vol 做验证

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 初始化模型
    model = BACFormer(num_classes=args.num_classes, img_size=args.img_size)

    # 开始训练
    train(model, train_loader, val_loader, args)

if __name__ == "__main__":
    main()