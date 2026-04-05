#!/bin/bash
echo "====================================="
echo "BACFormer 左心房测量工具 一键启动"
echo "====================================="

# 2. 自动检查并安装依赖（适配你的requirements路径）
echo "🔍 正在检查依赖安装情况..."
pip check > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "📦 检测到依赖未安装，正在自动安装所有依赖..."
    # 安装GPU版PyTorch
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    # 适配你的requirements.txt的实际路径
    pip install -r lists/lists_LeftAtrium/requirements.txt
    echo "✅ 依赖安装完成！"
fi

# 3. 准备权重目录
if [ ! -d "weights" ]; then
    mkdir weights
fi

# 4. 检查权重，没有的话自动训练
if [ -f "weights/best_model.pth" ]; then
    echo "✅ 检测到已训练好的权重，跳过训练..."
else
    echo "🚀 未找到权重，开始训练 50 轮..."
    python3 train.py
    echo "✅ 训练完成，权重已保存到 weights/best_model.pth"
fi

# 5. 根据系统决定是否启动GUI
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "====================================="
    echo "ℹ️  检测到Linux服务器环境（无头无图形界面）"
    echo "请将 weights/best_model.pth 下载到你的本地电脑"
    echo "然后在本地电脑启动GUI工具即可使用"
    echo "====================================="
else
    echo "🖥️  启动GUI工具..."
    python3 main.py
fi