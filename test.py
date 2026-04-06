import os
import sys
import torch
import numpy as np
import cv2
import warnings

# 忽略所有警告信息，保持输出整洁
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
    # 统计掩码中值为1的像素数量（左心房像素总数）
    la_pixels = np.sum(mask_2d == 1)
    # 计算面积：像素数乘以单个像素的面积（像素间距的平方）
    area = la_pixels * (pixel_spacing ** 2)

    # 将掩码转换为0-255的uint8格式，用于OpenCV处理
    mask_8u = (mask_2d * 255).astype(np.uint8)
    # 使用OpenCV查找外部轮廓，采用简单链近似方法压缩轮廓
    contours, _ = cv2.findContours(mask_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 初始化长轴长度为0
    long_axis_len = 0.0
    # 如果找到轮廓，继续处理
    if len(contours) > 0:
        # 找到面积最大的轮廓作为左心房边界
        la_contour = max(contours, key=cv2.contourArea)
        # 提取轮廓所有点的y坐标（第1列）
        y_coords = la_contour[:, 0, 1]
        # 提取轮廓所有点的x坐标（第0列）
        x_coords = la_contour[:, 0, 0]

        # 找到y坐标最小值的索引（最顶部的点）
        top_idx = np.argmin(y_coords)
        # 找到y坐标最大值的索引（最底部的点，二尖瓣环位置）
        bottom_idx = np.argmax(y_coords)
        # 获取顶部点的坐标(x, y)
        top_point = (x_coords[top_idx], y_coords[top_idx])
        # 获取底部点的坐标(x, y)
        bottom_point = (x_coords[bottom_idx], y_coords[bottom_idx])

        # 计算顶部点和底部点之间的欧氏距离（像素单位）
        pixel_dist = np.sqrt(((top_point[0] - bottom_point[0]) ** 2) + ((top_point[1] - bottom_point[1]) ** 2))
        # 将像素距离转换为实际毫米长度
        long_axis_len = pixel_dist * pixel_spacing

    # 返回四舍五入后的面积和长轴长度（保留两位小数）
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
    # 如果长轴长度或面积为0，无法计算容积，返回0
    if long_axis_len == 0 or area == 0:
        return 0.0

    # 使用单平面Simpson公式计算LAVmax（立方毫米）
    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    # 将立方毫米转换为毫升（1ml = 1000mm³）
    lavmax_ml = lavmax_mm3 / 1000
    # 返回四舍五入后的结果（保留两位小数）
    return round(lavmax_ml, 2)


def get_project_root():
    """
    自动定位项目根目录
    
    从当前文件所在目录向上查找，直到找到包含'lists'目录的位置
    
    Returns:
        str: 项目根目录路径
    """
    # 获取当前文件的绝对路径的目录部分
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 循环向上查找，直到找到包含'lists'子目录的位置
    while not os.path.exists(os.path.join(current_dir, "lists")):
        # 向上一级目录
        current_dir = os.path.dirname(current_dir)
    # 返回找到的项目根目录路径
    return current_dir


# 调用函数获取项目根目录并保存到全局变量
PROJECT_ROOT = get_project_root()

# 导入左心房数据集类
from datasets.dataset_left_atrium import Dataset_LeftAtrium


def main():
    """
    主函数：批量计算测试集样本的LAVmax值
    
    加载数据集，读取手动标注的GT掩码，计算几何参数并保存结果到CSV文件
    """
    # 创建测试数据集实例
    dataset = Dataset_LeftAtrium(
        root_path="lists",              # 数据根目录
        list_dir="lists/lists_LeftAtrium",  # 列表文件目录
        split="test",                   # 使用测试集
        img_size=256,                   # 图像尺寸
        norm_x=True                     # 启用归一化
    )

    # 构建结果保存目录路径
    RESULT_DIR = os.path.join(PROJECT_ROOT, "results/la_lavmax")
    # 创建目录（如果不存在）
    os.makedirs(RESULT_DIR, exist_ok=True)
    # 打开CSV文件准备写入结果
    result_file = open(os.path.join(RESULT_DIR, "lavmax_results.csv"), "w", encoding="utf-8")
    # 写入CSV表头
    result_file.write("样本ID,左心房面积(mm²),长轴长度(mm),LAVmax(ml)\n")

    # 获取测试集样本总数
    total_samples = len(dataset)
    # 初始化成功计数
    success_count = 0
    
    # 遍历所有样本
    for idx in range(total_samples):
        try:
            # 获取单个样本数据
            data = dataset[idx]
            # 提取样本ID
            sample_id = data['sample_id']
            # 提取像素间距
            pixel_spacing = data['pixel_spacing']
            # 提取GT掩码并转换为numpy数组（移除通道维度，转移到CPU）
            gt_mask = data['label'].squeeze(0).cpu().numpy()

            # 调用函数计算几何参数
            area, long_axis_len = calculate_la_geometry(gt_mask, pixel_spacing)
            # 调用函数计算LAVmax
            lavmax_ml = calculate_lavmax_single_plane(area, long_axis_len)

            # 成功计数加1
            success_count += 1
            # 打印进度和结果
            print(
                f"[{success_count}/{total_samples}] 📊 {sample_id} | 面积：{area} mm² | 长轴：{long_axis_len} mm | LAVmax：{lavmax_ml} ml")

            # 将结果写入CSV文件
            result_file.write(f"{sample_id},{area},{long_axis_len},{lavmax_ml}\n")

        except Exception as e:
            # 如果处理失败，打印错误信息（截取前50个字符）
            print(f"[{idx + 1}/{total_samples}] ❌ {dataset.samples[idx]} 处理失败：{str(e)[:50]}...")
            # 在CSV中标记为error
            result_file.write(f"{dataset.samples[idx]},error,error,error\n")
            # 继续处理下一个样本
            continue

    # 关闭CSV文件
    result_file.close()
    
    # 打印完成信息和统计结果
    print(f"\n🎉 计算完成！成功处理 {success_count}/{total_samples} 个样本")
    print(f"📁 结果文件保存至：{os.path.join(RESULT_DIR, 'lavmax_results.csv')}")


if __name__ == "__main__":
    # 程序入口：执行主函数
    main()