#!/bin/bash
echo "=============================================="
echo "BACFormer 左心房测量工具 【快速版】一键启动"
echo "先跑 50 轮（约1小时），拿到权重后直接开GUI"
echo "=============================================="
cd "$(dirname "$0")"

# 检查权重是否存在
if [ -f "weights/best_model.pth" ]; then
    echo "已找到权重，直接启动GUI..."
    python3 main.py
else
    echo "未找到权重，开始训练 50 轮..."
    python3 train.py --dataset LeftAtrium --root_path "$(pwd)" --max_epochs 50 --output_dir weights --img_size 224 --batch_size 4 --base_lr 0.05 --val_epoch 10 --num_classes 2
    echo "训练完成！启动GUI工具..."
    python3 main.py
fi