import cv2
import numpy as np


def calculate_la_geometry(mask_2d, pixel_spacing):
    """
    从左心房2D分割掩码计算面积和长轴长度
    
    Args:
        mask_2d (numpy.ndarray): 2D数组 (H, W)，1=左心房，0=背景
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
    单平面Simpson法估算左心房最大容积（LAVmax）
    
    临床标准公式：LAVmax = (8 × Area²) / (3 × π × L)
    
    Args:
        area (float): 左心房面积（mm²）
        long_axis_len (float): 长轴长度（mm）
        
    Returns:
        float: LAVmax（ml，1ml=1000mm³）
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


if __name__ == "__main__":
    # 创建一个256x256的全零掩码作为测试数据
    mask = np.zeros((256, 256))
    # 在中心区域设置一个100x100的正方形区域为左心房（值为1）
    mask[50:150, 50:150] = 1
    
    # 调用函数计算几何参数，像素间距设为0.5mm
    area, length = calculate_la_geometry(mask, pixel_spacing=0.5)
    # 调用函数计算LAVmax
    lavmax = calculate_lavmax_single_plane(area, length)
    
    # 打印测试结果
    print(f"✅ 测试结果：")
    print(f"左心房面积: {area} mm²")
    print(f"长轴长度: {length} mm")
    print(f"估算LAVmax: {lavmax} ml")