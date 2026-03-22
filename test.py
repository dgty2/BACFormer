import os
import sys
import torch
import numpy as np
import cv2
import warnings

warnings.filterwarnings("ignore")


# ========== LAVmax计算核心函数 ==========
def calculate_la_geometry(mask_2d, pixel_spacing):
    """从左心房2D分割掩码计算面积+长轴长度"""
    # 1. 计算左心房面积（像素数 × 像素间距²）
    la_pixels = np.sum(mask_2d == 1)  # 假设gt掩码中1=左心房，0=背景
    area = la_pixels * (pixel_spacing ** 2)

    # 2. 计算长轴长度（二尖瓣环中点→左心房顶部）
    mask_8u = (mask_2d * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_axis_len = 0.0
    if len(contours) > 0:
        # 取最大轮廓（左心房区域）
        la_contour = max(contours, key=cv2.contourArea)
        # 提取轮廓的x/y坐标
        y_coords = la_contour[:, 0, 1]
        x_coords = la_contour[:, 0, 0]

        # 左心房顶部（y最小）和底部（y最大，二尖瓣环位置）
        top_idx = np.argmin(y_coords)
        bottom_idx = np.argmax(y_coords)
        top_point = (x_coords[top_idx], y_coords[top_idx])
        bottom_point = (x_coords[bottom_idx], y_coords[bottom_idx])

        # 计算像素距离 → 转换为毫米
        pixel_dist = np.sqrt(((top_point[0] - bottom_point[0]) ** 2) + ((top_point[1] - bottom_point[1]) ** 2))
        long_axis_len = pixel_dist * pixel_spacing

    return round(area, 2), round(long_axis_len, 2)


def calculate_lavmax_single_plane(area, long_axis_len):
    """单平面Simpson法估算左心房最大容积（LAVmax）"""
    if long_axis_len == 0 or area == 0:
        return 0.0

    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    lavmax_ml = lavmax_mm3 / 1000  # 转换为临床常用单位
    return round(lavmax_ml, 2)


# ========================================

# ========== 自动定位项目根目录 ==========
def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(current_dir, "lists")):
        current_dir = os.path.dirname(current_dir)
    return current_dir


PROJECT_ROOT = get_project_root()
# ========================================

# 导入数据加载模块
from datasets.dataset_left_atrium import Dataset_LeftAtrium


# 主函数（仅用手动标注掩码计算）
def main():
    # 1. 初始化数据集（加载影像+手动gt掩码）
    dataset = Dataset_LeftAtrium(
        root_path="lists",
        list_dir="lists/lists_LeftAtrium",
        split="test",
        img_size=256,
        norm_x=True
    )

    # 2. 创建结果保存目录
    RESULT_DIR = os.path.join(PROJECT_ROOT, "results/la_lavmax")
    os.makedirs(RESULT_DIR, exist_ok=True)
    result_file = open(os.path.join(RESULT_DIR, "lavmax_results.csv"), "w", encoding="utf-8")
    result_file.write("样本ID,左心房面积(mm²),长轴长度(mm),LAVmax(ml)\n")

    # 3. 批量处理1-100号样本（仅用手动gt掩码）
    total_samples = len(dataset)
    success_count = 0
    for idx in range(total_samples):
        try:
            # 加载单样本（包含影像+gt掩码）
            data = dataset[idx]
            sample_id = data['sample_id']
            pixel_spacing = data['pixel_spacing']
            # 提取手动标注的gt掩码（2D）
            gt_mask = data['label'].squeeze(0).cpu().numpy()  # (H,W)，1=左心房，0=背景

            # 核心：用手动gt掩码计算LAVmax（完全不用模型/权重）
            area, long_axis_len = calculate_la_geometry(gt_mask, pixel_spacing)
            lavmax_ml = calculate_lavmax_single_plane(area, long_axis_len)

            # 打印进度（非0数值，真实结果）
            success_count += 1
            print(
                f"[{success_count}/{total_samples}] 📊 {sample_id} | 面积：{area} mm² | 长轴：{long_axis_len} mm | LAVmax：{lavmax_ml} ml")

            # 保存结果
            result_file.write(f"{sample_id},{area},{long_axis_len},{lavmax_ml}\n")

        except Exception as e:
            print(f"[{idx + 1}/{total_samples}] ❌ {dataset.samples[idx]} 处理失败：{str(e)[:50]}...")
            result_file.write(f"{dataset.samples[idx]},error,error,error\n")
            continue

    # 收尾
    result_file.close()
    print(f"\n🎉 计算完成！成功处理 {success_count}/{total_samples} 个样本")
    print(f"📁 结果文件保存至：{os.path.join(RESULT_DIR, 'lavmax_results.csv')}")


if __name__ == "__main__":
    main()