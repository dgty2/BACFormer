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
    # 根据CUDA可用性自动选择设备（GPU优先，否则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建BACFormer模型实例，设置分类数为2（背景+左心房）
    model = BACFormer(num_classes=2)
    # 从指定路径加载模型权重，并映射到对应设备
    model.load_state_dict(torch.load(weight_path, map_location=device))
    # 将模型移动到指定设备（CPU或GPU）
    model.to(device)
    # 将模型设置为评估模式，关闭dropout和batchnorm的训练行为
    model.eval()
    # 返回加载好权重的模型和设备信息
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
    # 复制输入图像数据，避免修改原始数据
    img = img_data.copy()
    # 将图像缩放到224x224像素，使用线性插值保持图像质量
    img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # 将numpy数组转换为PyTorch张量，添加批次维度和通道维度，并移动到指定设备
    tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    # 禁用梯度计算，减少内存占用并加速推理
    with torch.no_grad():
        # 执行模型前向传播，得到输出logits
        output = model(tensor)
    
    # 对输出应用softmax得到概率分布，取最大概率的类别索引，移除多余维度并转回numpy数组
    pred = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
    # 将预测结果二值化：类别1（左心房）设为1，其他设为0，并转换为uint8类型
    pred = (pred == 1).astype(np.uint8)
    
    # 针对镜像影像进行左侧截断：去掉左侧84列（约37.5%），只保留右侧140列的左心房区域
    pred = pred[:, 84:]
    
    # 创建3x3椭圆形结构元素，用于形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # 执行形态学开运算（先腐蚀后膨胀），去除小噪点和平滑边界
    pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
    
    # 使用scipy进行连通域分析，标记所有连通的区域
    labeled, n_labels = label(pred)
    # 如果存在连通域，则筛选出最大的连通域作为左心房
    if n_labels > 0:
        # 计算每个连通域的像素数量，生成大小列表
        component_sizes = [sum_labels(pred == i, labeled) for i in range(1, n_labels + 1)]
        # 找到像素数最多的连通域的标签（索引+1，因为标签从1开始）
        max_label = component_sizes.index(max(component_sizes)) + 1
        # 只保留最大连通域，其他区域全部置0
        pred = (labeled == max_label).astype(np.uint8)
    
    # 创建一个224x224的全零掩码作为完整尺寸的画布
    full_pred = np.zeros((224, 224), dtype=np.uint8)
    # 将裁剪后的预测结果放回画布的右侧区域（从第84列开始）
    full_pred[:, 84:] = pred
    # 更新pred为恢复后的完整掩码
    pred = full_pred
    
    # 计算原始图像的总像素数
    total_pixels = img_data.shape[0] * img_data.shape[1]
    # 设置允许的最大左心房像素数（不超过总面积的1/5）
    max_allowed = total_pixels // 5
    # 如果预测的左心房面积超过限制，则进行裁剪
    if np.sum(pred) > max_allowed:
        # 将掩码展平为一维数组
        pred_flat = pred.flatten()
        # 找到所有值为1的像素的索引位置
        ones_indices = np.where(pred_flat == 1)[0]
        # 如果前景像素数量超过限制
        if len(ones_indices) > max_allowed:
            # 将超出限制的像素置为0（保留前max_allowed个像素）
            pred_flat[ones_indices[max_allowed:]] = 0
            # 将一维数组恢复为原始二维形状
            pred = pred_flat.reshape(pred.shape)
    
    # 将掩码缩放回原始图像尺寸，使用最近邻插值保持二值特性
    pred = cv2.resize(pred, (img_data.shape[1], img_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 打印调试信息：左心房像素数量和掩码形状
    print(f"[分割调试] 左心房像素数：{np.sum(pred)}, mask形状：{pred.shape}")
    # 返回最终的二值分割掩码
    return pred
