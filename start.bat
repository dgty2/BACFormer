@echo off
echo =====================================
echo BACFormer 左心房测量工具 一键启动
echo =====================================
cd /d %~dp0

:: 2. 自动检查并安装依赖
echo 🔍 正在检查依赖安装情况...
pip check >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 检测到依赖未安装，正在自动安装所有依赖...
    :: 安装GPU版PyTorch
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    :: 安装requirements
    pip install -r lists/lists_LeftAtrium/requirements.txt
    echo ✅ 依赖安装完成！
)

:: 3. 准备权重目录
if not exist "weights" (
    mkdir weights
)

:: 4. 检查权重，没有的话自动训练
if exist "weights\best_model.pth" (
    echo ✅ 检测到已训练好的权重，跳过训练...
) else (
    echo 🚀 未找到权重，开始训练 50 轮...
    python train.py
    echo ✅ 训练完成，权重已保存到 weights/best_model.pth
)

:: 5. 启动GUI工具
echo.
echo 🖥️  启动GUI工具...
python main.py

pause