import torch
import os
import numpy as np
from networks.BACFormer import BACFormer

def load_bacformer_model(weight_path, num_classes=2):
    """加载 BACFormer 模型（左心房分割：背景+左心房=2类）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BACFormer(num_classes=num_classes).to(device)

    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        # 兼容不同权重保存格式
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ 模型权重加载成功：{weight_path}")
    else:
        raise FileNotFoundError(f"❌ 权重文件不存在：{weight_path}")

    return model, device

def predict_mask(model, device, slice_data):
    """对单张 2D 切片推理左心房掩码（输出 H×W，1=左心房，0=背景）"""
    # 数据归一化 + 转 Tensor
    slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
    slice_tensor = torch.from_numpy(slice_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    # 无梯度推理
    with torch.no_grad():
        pred = model(slice_tensor)
        mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()  # (H, W)

    return mask