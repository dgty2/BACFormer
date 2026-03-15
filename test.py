#!/usr/bin/env python
# BACFormer 左心房分割测试脚本
import argparse
import torch
from torch.utils.data import DataLoader
from datasets.dataset_synapse import SynapseDataset
from networks.BACFormer import BACFormer
from utils import calculate_dice
import os

def main():
    parser = argparse.ArgumentParser(description="BACFormer 左心房分割测试")
    parser.add_argument("--root_path", default="./data/Synapse", type=str)
    parser.add_argument("--list_dir", default="./lists/lists_Synapse", type=str)
    parser.add_argument("--model_path", default="./model_out/best_la_model.pth", type=str)
    parser.add_argument("--num_classes", default=1, type=int)
    args = parser.parse_args()

    # 加载测试集
    test_dataset = SynapseDataset(args.root_path, args.list_dir, split="test_vol")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 加载模型
    model = BACFormer(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 计算测试集Dice
    total_dice = 0.0
    with torch.no_grad():
        for batch in test_loader:
            img, label = batch["image"].to(device), batch["label"].to(device)
            output = model(img)
            total_dice += calculate_dice(output, label)

    avg_dice = total_dice / len(test_loader)
    print("=" * 50)
    print(f"📊 测试集最终Dice系数: {avg_dice:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()