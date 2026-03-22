@echo off
echo ==============================================
echo BACFormer 左心房测量工具 一键启动脚本
echo 规则：有权重直接开GUI，无权重自动训练
echo ==============================================
cd /d %~dp0

:: 检查权重是否存在
if exist "weights\best_model.pth" (
    echo 已找到最优权重，直接启动GUI...
    python main.py
) else (
    echo 未找到权重，开始自动训练模型...
    echo 训练时长：显卡≈30分钟 | CPU≈5小时
    python train.py --dataset LeftAtrium --root_path %~dp0 --max_epochs 50 --output_dir weights --img_size 224 --batch_size 4 --base_lr 0.05 --val_epoch 10 --num_classes 2
    echo 训练完成！启动GUI工具...
    python main.py
)

pause