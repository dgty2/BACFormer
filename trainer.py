#!/usr/bin/env python
# BACFormer 左心房分割训练器
import torch
from utils import DiceBoundaryLoss, calculate_dice

# 初始化损失函数
criterion = DiceBoundaryLoss()

def train_one_epoch(model, loader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for batch in loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += calculate_dice(output, label)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    return avg_loss, avg_dice

def validate_one_epoch(model, loader, device):
    """验证一个epoch"""
    model.eval()
    total_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            img, label = batch["image"].to(device), batch["label"].to(device)
            output = model(img)
            total_dice += calculate_dice(output, label)

    avg_dice = total_dice / len(loader)
    return avg_dice

def train(model, train_loader, val_loader, args):
    """完整训练流程"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    best_dice = 0.0

    print("=" * 50)
    print("🚀 开始左心房分割训练")
    print(f"设备: {device} | 类别数: {args.num_classes} | 轮数: {args.max_epochs}")
    print("=" * 50)

    for epoch in range(args.max_epochs):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = validate_one_epoch(model, val_loader, device)

        # 保存最优模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), f"{args.save_path}/best_la_model.pth")
            print(f"✅ 新最优模型保存！Val Dice: {val_dice:.4f}")

        # 打印训练日志
        print(f"Epoch [{epoch+1}/{args.max_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Dice: {train_dice:.4f} | "
              f"Val Dice: {val_dice:.4f}")

    print("=" * 50)
    print(f"🏁 训练完成！最优验证集Dice: {best_dice:.4f}")
    print("=" * 50)