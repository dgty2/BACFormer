import os
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtGui import QImage, QPixmap


def load_nii(file_path):
    """
    加载NIfTI格式的医学影像文件
    
    Args:
        file_path (str): NII文件路径
        
    Returns:
        tuple: (data, pixdim)
            - data: 影像数据（numpy数组）
            - pixdim: 像素间距信息（前4个维度）
    """
    nii = nib.load(file_path)
    data = nii.get_fdata()
    pixdim = nii.header['pixdim'][:4]
    return data, pixdim


def load_label_mask(img_path):
    """
    自动加载与影像文件对应的手动标注掩码（GT）
    
    根据影像文件路径自动推导对应的label文件路径
    支持多种命名格式（_gt.nii.gz 或 _gt.nii）
    
    Args:
        img_path (str): 影像文件路径
        
    Returns:
        numpy.ndarray: GT掩码数据（uint8类型）
        
    Raises:
        FileNotFoundError: 当掩码文件不存在时抛出
    """
    base_name = os.path.basename(img_path)
    case_id = base_name.replace(".nii.gz", "").replace(".nii", "")
    
    img_dir = os.path.dirname(img_path)
    lists_dir = os.path.dirname(img_dir)
    label_dir = os.path.join(lists_dir, "label")
    
    label_path = os.path.join(label_dir, f"{case_id}_gt.nii.gz")
    if not os.path.exists(label_path):
        label_path = os.path.join(label_dir, f"{case_id}_gt.nii")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"掩码文件不存在：{label_path}")
    
    mask = nib.load(label_path).get_fdata()
    return mask.astype(np.uint8)


def slice_to_qimage(slice_data, mask=None, alpha=0.3):
    """
    将2D切片数据和掩码转换为QPixmap用于Qt界面显示
    
    支持灰度图像归一化和红色半透明掩码叠加
    针对镜像影像进行左侧截断过滤
    
    Args:
        slice_data (numpy.ndarray): 2D切片数据
        mask (numpy.ndarray, optional): 2D掩码数据
        alpha (float): 掩码透明度，默认0.3
        
    Returns:
        QPixmap: 可用于Qt Label显示的图像
    """
    slice_norm = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(slice_norm).convert("RGB")
    
    if mask is not None and np.sum(mask) > 0:
        w = mask.shape[1]
        mask[:, :int(w*0.375)] = 0
        mask = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, "L")
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, int(255 * alpha)))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    
    img_np = np.array(img)
    h, w, ch = img_np.shape
    bytes_per_line = ch * w
    q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)


def export_excel_report(case_id, area, long_axis, lavmax, dice, save_path):
    """
    导出测量结果到Excel报告文件
    
    Args:
        case_id (str): 病例ID
        area (float): 左心房面积（mm²）
        long_axis (float): 长轴长度（mm）
        lavmax (float): LAVmax容积（ml）
        dice (float or None): Dice系数，可为None
        save_path (str): 报告保存目录路径
        
    Returns:
        str: Excel文件的完整路径
    """
    import pandas as pd
    data = {
        "病例ID": [case_id],
        "左心房面积(mm²)": [area],
        "长轴长度(mm)": [long_axis],
        "LAVmax(ml)": [lavmax],
        "Dice系数": [dice if dice is not None else "无"]
    }
    df = pd.DataFrame(data)
    excel_path = os.path.join(save_path, f"{case_id}_测量报告.xlsx")
    df.to_excel(excel_path, index=False)
    return excel_path
