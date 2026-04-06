import os
import sys
import torch
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")


def calculate_la_geometry(mask_2d, pixel_spacing):
    """
    从左心房2D分割掩码计算面积和长轴长度
    
    Args:
        mask_2d (numpy.ndarray): 2D数组，1表示左心房，0表示背景
        pixel_spacing (float): 像素间距（mm/像素）
        
    Returns:
        tuple: (area, long_axis_len)
            - area: 左心房面积（mm²）
            - long_axis_len: 长轴长度（mm）
    """
    la_pixels = np.sum(mask_2d == 1)
    area = la_pixels * (pixel_spacing ** 2)

    mask_8u = (mask_2d * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_axis_len = 0.0
    if len(contours) > 0:
        la_contour = max(contours, key=cv2.contourArea)
        y_coords = la_contour[:, 0, 1]
        x_coords = la_contour[:, 0, 0]

        top_idx = np.argmin(y_coords)
        bottom_idx = np.argmax(y_coords)
        top_point = (x_coords[top_idx], y_coords[top_idx])
        bottom_point = (x_coords[bottom_idx], y_coords[bottom_idx])

        pixel_dist = np.sqrt(((top_point[0] - bottom_point[0]) ** 2) + ((top_point[1] - bottom_point[1]) ** 2))
        long_axis_len = pixel_dist * pixel_spacing

    return round(area, 2), round(long_axis_len, 2)


def calculate_lavmax_single_plane(area, long_axis_len):
    """
    使用单平面Simpson法估算左心房最大容积（LAVmax）
    
    临床标准公式：LAVmax = (8 × Area²) / (3 × π × L)
    
    Args:
        area (float): 左心房面积（mm²）
        long_axis_len (float): 长轴长度（mm）
        
    Returns:
        float: LAVmax（ml），如果输入为0则返回0.0
    """
    if long_axis_len == 0 or area == 0:
        return 0.0

    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    lavmax_ml = lavmax_mm3 / 1000
    return round(lavmax_ml, 2)


def get_project_root():
    """
    自动定位项目根目录
    
    从当前文件所在目录向上查找，直到找到包含'lists'目录的位置
    
    Returns:
        str: 项目根目录路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "lists")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


PROJECT_ROOT = get_project_root()

from datasets.dataset_left_atrium import Dataset_LeftAtrium


def main():
    """
    主函数：批量计算测试集样本的LAVmax值
    
    加载数据集，读取手动标注的GT掩码，计算几何参数并保存结果到CSV文件
    """
    dataset = Dataset_LeftAtrium(
        root_path="lists",
        list_dir="lists/lists_LeftAtrium",
        split="test",
        img_size=256,
        norm_x=True
    )

    RESULT_DIR = os.path.join(PROJECT_ROOT, "results/la_lavmax")
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file = open(os.path.join(RESULT_DIR, "lavmax_results.csv"), "w", encoding="utf-8")
    result_file.write("样本ID,左心房面积(mm²),长轴长度(mm),LAVmax(ml)\n")

    total_samples = len(dataset)
    success_count = 0
    for idx in range(total_samples):
        try:
            data = dataset[idx]
            sample_id = data['sample_id']
            pixel_spacing = data['pixel_spacing']
            gt_mask = data['label'].squeeze(0).cpu().numpy()

            area, long_axis_len = calculate_la_geometry(gt_mask, pixel_spacing)
            lavmax_ml = calculate_lavmax_single_plane(area, long_axis_len)

            success_count += 1
            print(
                f"[{success_count}/{total_samples}] 📊 {sample_id} | 面积：{area} mm² | 长轴：{long_axis_len} mm | LAVmax：{lavmax_ml} ml")

            result_file.write(f"{sample_id},{area},{long_axis_len},{lavmax_ml}\n")

        except Exception as e:
            print(f"[{idx + 1}/{total_samples}] ❌ {dataset.samples[idx]} 处理失败：{str(e)[:50]}...")
            result_file.write(f"{dataset.samples[idx]},error,error,error\n")
            continue

    result_file.close()
    print(f"\n🎉 计算完成！成功处理 {success_count}/{total_samples} 个样本")
    print(f"📁 结果文件保存至：{os.path.join(RESULT_DIR, 'lavmax_results.csv')}")


if __name__ == "__main__":
    main()