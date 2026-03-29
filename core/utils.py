import os
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from PyQt5.QtGui import QImage, QPixmap
def load_nii(file_path):
    """加载NII文件，返回影像数据+像素间距"""
    nii = nib.load(file_path)
    data = nii.get_fdata()
    pixdim = nii.header['pixdim'][:4]  # 取前4个像素间距
    return data, pixdim
def load_label_mask(img_path):
    """
    自动匹配手动掩码GT，修复路径重复问题
    :param img_path: 影像文件路径
    :return: GT掩码数据
    """
    # 提取文件名
    base_name = os.path.basename(img_path)
    case_id = base_name.replace(".nii.gz", "").replace(".nii", "")
    # 正确拼接label路径（绝对路径，无重复）
    img_dir = os.path.dirname(img_path)
    lists_dir = os.path.dirname(img_dir)
    label_dir = os.path.join(lists_dir, "label")
    # 匹配你的文件命名
    label_path = os.path.join(label_dir, f"{case_id}_gt.nii.gz")
    if not os.path.exists(label_path):
        label_path = os.path.join(label_dir, f"{case_id}_gt.nii")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"掩码文件不存在：{label_path}")
    mask = nib.load(label_path).get_fdata()
    return mask.astype(np.uint8)
def slice_to_qimage(slice_data, mask=None, alpha=0.3):
    """将2D切片+mask转换为QPixmap，同步调整右侧截断比例"""
    # 归一化到0-255
    slice_norm = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
    img = Image.fromarray(slice_norm).convert("RGB")
    # 叠加mask，同步调整右侧截断到62.5%位置
    if mask is not None and np.sum(mask) > 0:
        w = mask.shape[1]
        mask[:, int(w*0.625):] = 0  # 只截断最右侧的左心室部分
        mask = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask, "L")
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, int(255 * alpha)))
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    # 转换为Qt图像
    img_np = np.array(img)
    h, w, ch = img_np.shape
    bytes_per_line = ch * w
    q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_img)
def export_excel_report(case_id, area, long_axis, lavmax, dice, save_path):
    """导出Excel报告（简化版）"""
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