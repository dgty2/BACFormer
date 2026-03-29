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
    微调版：放宽右侧截断阈值，保留更多左心房区域
    """
    # 1. 预处理
    img = img_data.copy()
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # 2. 模型推理
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    # 3. 类别修正
    pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
    pred = (pred == 1).astype(np.uint8)
    # 4. 调整右侧截断阈值：从112（50%）改为140（约62.5%），保留更多左心房
    pred = pred[:, :140]  # 只截断更靠右的部分，保留左心房右侧
    # 5. 形态学去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    # 6. 保留最大的连通域
    labeled, n_labels = label(pred)
    if n_labels > 0:
        component_sizes = [sum_labels(pred == i, labeled) for i in range(1, n_labels + 1)]
        max_label = component_sizes.index(max(component_sizes)) + 1
        pred = (labeled == max_label).astype(np.uint8)
    # 7. 恢复尺寸，同步调整右侧截断位置
    full_pred = np.zeros((224, 224), dtype=np.uint8)
    full_pred[:, :140] = pred
    pred = full_pred
    # 8. 异常面积限制
    total_pixels = img_data.shape[0] * img_data.shape[1]
    max_allowed = total_pixels // 5
    if np.sum(pred) > max_allowed:
        pred_flat = pred.flatten()
        ones_indices = np.where(pred_flat == 1)[0]
        if len(ones_indices) > max_allowed:
            pred_flat[ones_indices[max_allowed:]] = 0
            pred = pred_flat.reshape(pred.shape)
    # 9. 恢复原图尺寸
    pred = cv2.resize(pred, (img_data.shape[1], img_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    print(f"[分割调试] 左心房像素数：{np.sum(pred)}, mask形状：{pred.shape}")
    return pred