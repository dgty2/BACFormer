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
    # 使用nibabel库加载NIfTI格式的医学影像文件
    nii = nib.load(file_path)
    # 提取影像的数值数据，转换为numpy数组
    data = nii.get_fdata()
    # 从文件头中提取像素间距信息（前4个维度：x、y、z方向和体素深度）
    pixdim = nii.header['pixdim'][:4]
    # 返回影像数据和像素间距元组
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
    # 提取文件名的基本部分（包含扩展名）
    base_name = os.path.basename(img_path)
    # 移除扩展名获取病例ID，支持.nii.gz和.nii两种格式
    case_id = base_name.replace(".nii.gz", "").replace(".nii", "")

    # 获取影像文件所在的目录路径
    img_dir = os.path.dirname(img_path)
    # 向上一级目录获取lists文件夹路径
    lists_dir = os.path.dirname(img_dir)
    # 拼接label子目录路径
    label_dir = os.path.join(lists_dir, "label")

    # 构建首选的掩码文件路径（.nii.gz格式）
    label_path = os.path.join(label_dir, f"{case_id}_gt.nii.gz")
    # 如果首选路径不存在，尝试备选格式
    if not os.path.exists(label_path):
        # 尝试.nii格式的掩码文件
        label_path = os.path.join(label_dir, f"{case_id}_gt.nii")
    # 如果两种格式都不存在，抛出文件未找到异常
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"掩码文件不存在：{label_path}")

    # 加载掩码文件并提取数值数据
    mask = nib.load(label_path).get_fdata()
    # 将掩码数据转换为uint8类型（0-255整数）并返回
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
    # 对切片数据进行归一化处理：缩放到0-255范围，添加小值1e-8防止除零
    slice_norm = ((slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8) * 255).astype(np.uint8)
    # 将numpy数组转换为PIL灰度图像，再转为RGB模式
    img = Image.fromarray(slice_norm).convert("RGB")

    # 如果提供了掩码且掩码非空（至少有一个非零像素），则叠加显示
    if mask is not None and np.sum(mask) > 0:
        # 获取掩码图像的宽度
        w = mask.shape[1]
        # 针对镜像影像进行左侧截断：清除左侧37.5%区域（左心室误标部分）
        mask[:, :int(w * 0.375)] = 0
        # 将掩码二值化并转换为0-255范围（大于0的设为255，否则为0）
        mask = (mask > 0).astype(np.uint8) * 255
        # 将掩码数组转换为PIL灰度图像（"L"模式表示8位灰度）
        mask_img = Image.fromarray(mask, "L")
        # 创建一个与原图相同大小的RGBA透明图层，初始化为完全透明
        overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
        # 创建绘图对象用于在透明图层上绘制
        draw = ImageDraw.Draw(overlay)
        # 将掩码作为位图绘制到透明图层上，使用红色填充并设置透明度
        draw.bitmap((0, 0), mask_img, fill=(255, 0, 0, int(255 * alpha)))
        # 将原图和红色掩码图层进行Alpha混合，然后转回RGB模式
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # 将PIL图像转换回numpy数组，用于Qt处理
    img_np = np.array(img)
    # 获取图像的高度、宽度和通道数
    h, w, ch = img_np.shape
    # 计算每行的字节数（宽度 × 通道数）
    bytes_per_line = ch * w
    # 创建QImage对象，指定数据指针、尺寸、步长和像素格式
    q_img = QImage(img_np.data, w, h, bytes_per_line, QImage.Format_RGB888)
    # 将QImage转换为QPixmap并返回，用于Qt界面显示
    return QPixmap.fromImage(q_img)


def export_excel_report(case_id, area, long_axis, lavmax, dice, save_path, lavmin=None, ef=None):
    """
    导出测量结果到Excel报告文件

    Args:
        case_id (str): 病例ID
        area (float): 左心房面积（mm²）
        long_axis (float): 长轴长度（mm）
        lavmax (float): LAVmax容积（ml）
        dice (float or None): Dice系数，可为None
        save_path (str): 报告保存目录路径
        lavmin (float or None): LAVmin容积（ml），可为None
        ef (float or None): 射血分数EF（%），可为None

    Returns:
        str: Excel文件的完整路径
    """
    # 导入pandas库用于Excel文件操作
    import pandas as pd
    # 构建测量结果数据字典，每个字段为列表形式（一行数据）
    data = {
        "病例ID": [case_id],
        "左心房面积(mm²)": [area if area is not None else "无"],
        "长轴长度(mm)": [long_axis if long_axis is not None else "无"],
        "LAVmax(ml)": [lavmax if lavmax is not None else "无"],
        "LAVmin(ml)": [lavmin if lavmin is not None else "无"],
        "射血分数EF(%)": [ef if ef is not None else "无"],
        "Dice系数": [dice if dice is not None else "无"]
    }
    # 将字典转换为pandas DataFrame表格对象
    df = pd.DataFrame(data)
    # 拼接完整的Excel文件路径，文件名包含病例ID
    excel_path = os.path.join(save_path, f"{case_id}_测量报告.xlsx")
    # 将DataFrame写入Excel文件，不包含索引列
    df.to_excel(excel_path, index=False)
    # 返回生成的Excel文件完整路径
    return excel_path