import torch
import numpy as np
import cv2
import torch.nn.functional as F
from networks.BACFormer import BACFormer
from scipy.ndimage import label, sum_labels


def load_bacformer_model(weight_path):
    """
    加载BACFormer模型及其权重
    
    Args:
        weight_path (str): 模型权重文件路径
        
    Returns:
        tuple: (model, device)
            - model: 加载权重后的模型（eval模式）
            - device: 模型所在的设备（CPU或GPU）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BACFormer(num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_mask(model, device, img_data):
    """
    使用模型预测单张图像的分割掩码
    
    针对镜像影像进行左侧截断处理，自动定位右侧的左心房区域
    包含形态学去噪、连通域分析等后处理步骤
    
    Args:
        model (BACFormer): 训练好的分割模型
        device (torch.device): 模型设备
        img_data (numpy.ndarray): 输入图像数据（2D）
        
    Returns:
        numpy.ndarray: 预测的二值掩码（与原图同尺寸）
    """
    img = img_data.copy()
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    
    pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
    pred = (pred == 1).astype(np.uint8)
    
    pred = pred[:, 84:]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    
    labeled, n_labels = label(pred)
    if n_labels > 0:
        component_sizes = [sum_labels(pred == i, labeled) for i in range(1, n_labels + 1)]
        max_label = component_sizes.index(max(component_sizes)) + 1
        pred = (labeled == max_label).astype(np.uint8)
    
    full_pred = np.zeros((224, 224), dtype=np.uint8)
    full_pred[:, 84:] = pred
    pred = full_pred
    
    total_pixels = img_data.shape[0] * img_data.shape[1]
    max_allowed = total_pixels // 5
    if np.sum(pred) > max_allowed:
        pred_flat = pred.flatten()
        ones_indices = np.where(pred_flat == 1)[0]
        if len(ones_indices) > max_allowed:
            pred_flat[ones_indices[max_allowed:]] = 0
            pred = pred_flat.reshape(pred.shape)
    
    pred = cv2.resize(pred, (img_data.shape[1], img_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    print(f"[分割调试] 左心房像素数：{np.sum(pred)}, mask形状：{pred.shape}")
    return pred
