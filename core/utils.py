import os
import nibabel as nib
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
import cv2
import pandas as pd


def load_nii(file_path):
    """读取 .nii/.nii.gz 影像，返回数据数组 + 像素间距（mm）"""
    img = nib.load(file_path)
    data = img.get_fdata()
    pixdim = img.header["pixdim"][1]  # 取第一个维度的像素间距
    return data, pixdim


def load_label_mask(img_path):
    """
    从影像路径加载对应的手动掩码（gt）
    :param img_path: 影像文件路径（示例：.../lists/img/patient0004_4CH_ES.nii.gz）
    :return: 掩码 numpy 数组
    """
    # 1. 提取病例 ID（去掉 .nii.gz/.nii 后缀）
    base_name = os.path.basename(img_path)
    case_id = base_name.replace(".nii.gz", "").replace(".nii", "")

    # 2. 定位 label 目录（从 img 目录回退到 lists 目录，再进入 label）
    img_dir = os.path.dirname(img_path)  # .../lists/img
    lists_dir = os.path.dirname(img_dir)  # .../lists
    label_dir = os.path.join(lists_dir, "label")  # .../lists/label

    # 3. 拼接正确的掩码路径（和你的文件结构完全匹配）
    label_path = os.path.join(label_dir, f"{case_id}_gt.nii.gz")

    # 4. 检查文件是否存在
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"未找到对应掩码：{label_path}")

    # 5. 加载并返回掩码数据
    mask_nii = nib.load(label_path)
    mask_data = mask_nii.get_fdata()
    return mask_data.astype(np.int32)


def slice_to_qimage(slice_data, mask=None):
    """将 2D 切片转 QPixmap，可选叠加掩码（半透明绿色）【已修复报错】"""
    # 归一化到 0-255
    slice_norm = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255
    slice_uint8 = slice_norm.astype(np.uint8)

    if mask is not None:
        # 转彩色并叠加掩码
        slice_color = cv2.cvtColor(slice_uint8, cv2.COLOR_GRAY2BGR)
        mask_color = np.zeros_like(slice_color)
        mask_color[mask == 1] = [0, 255, 0]  # 绿色标记左心房
        slice_color = cv2.addWeighted(slice_color, 0.7, mask_color, 0.3, 0)
        slice_uint8 = cv2.cvtColor(slice_color, cv2.COLOR_BGR2RGB)

        # 修复核心：tobytes() 替代 data，解决QImage报错
        q_img = QImage(
            slice_uint8.tobytes(),
            slice_uint8.shape[1],
            slice_uint8.shape[0],
            slice_uint8.strides[0],
            QImage.Format_RGB888
        )
    else:
        # 修复核心：tobytes() 替代 data，解决QImage报错
        q_img = QImage(
            slice_uint8.tobytes(),
            slice_uint8.shape[1],
            slice_uint8.shape[0],
            QImage.Format_Grayscale8
        )

    return QPixmap.fromImage(q_img)


def export_excel_report(case_id, area, long_axis, lavmax, dice=None, save_path="./results"):
    """导出 Excel 测量报告"""
    os.makedirs(save_path, exist_ok=True)
    report_data = {
        "病例ID": [case_id],
        "左心房面积(mm²)": [area],
        "长轴长度(mm)": [long_axis],
        "LAVmax(ml)": [lavmax],
        "Dice系数": [dice if dice else "无手动掩码"]
    }
    df = pd.DataFrame(report_data)
    excel_path = os.path.join(save_path, f"{case_id}_report.xlsx")
    df.to_excel(excel_path, index=False)
    return excel_path