import torch
import numpy as np
import cv2
import torch.nn.functional as F
from networks.BACFormer import BACFormer
from scipy.ndimage import label, sum_labels


def load_bacformer_model(weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BACFormer(num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def predict_mask(model, device, img_data):
    """
    修复版：
    1. 修复类别反转问题：左心房=1（训练标签定义：0=背景，1=前景）
    2. 修复连通域选择：只保留最大的前景连通域（左心房），过滤噪声
    """
    # 1. 预处理：和训练完全一致！删掉了多余的min-max归一化！
    img = img_data.copy()
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 2. 模型推理
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)

    # 3. 类别修正：0=背景，1=左心房
    pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
    pred = (pred == 1).astype(np.uint8)

    # 4. 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)

    # 5. 保留最大连通域（左心房主体）
    labeled, n_labels = label(pred)
    if n_labels > 0:
        component_sizes = [sum_labels(pred == i, labeled) for i in range(1, n_labels + 1)]
        max_label = component_sizes.index(max(component_sizes)) + 1
        pred = (labeled == max_label).astype(np.uint8)

    # 6. 异常面积限制（兼容原有逻辑）
    total_pixels = img_data.shape[0] * img_data.shape[1]
    max_allowed = total_pixels // 5
    if np.sum(pred) > max_allowed:
        pred_flat = pred.flatten()
        ones_indices = np.where(pred_flat == 1)[0]
        if len(ones_indices) > max_allowed:
            pred_flat[ones_indices[max_allowed:]] = 0
            pred = pred_flat.reshape(pred.shape)

    # 7. 恢复尺寸
    pred = cv2.resize(pred, (img_data.shape[1], img_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    print(f"[分割调试] 左心房像素数：{np.sum(pred)}, mask形状：{pred.shape}")
    return pred