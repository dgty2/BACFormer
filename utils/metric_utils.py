import cv2
import numpy as np


def calculate_la_geometry(mask_2d, pixel_spacing):
    """
    从左心房2D分割掩码计算面积+长轴长度
    :param mask_2d: 2D数组 (H, W)，1=左心房，0=背景
    :param pixel_spacing: 像素间距（mm/像素）
    :return: area（mm²）, long_axis_len（mm）
    """
    # 1. 计算左心房面积（像素数 × 像素间距²）
    la_pixels = np.sum(mask_2d == 1)
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
    """
    单平面Simpson法估算左心房最大容积（LAVmax）
    临床标准公式：LAVmax = (8 × Area²) / (3 × π × L)
    :return: LAVmax（ml，1ml=1000mm³）
    """
    if long_axis_len == 0 or area == 0:
        return 0.0

    lavmax_mm3 = (8 * (area ** 2)) / (3 * np.pi * long_axis_len)
    lavmax_ml = lavmax_mm3 / 1000  # 转换为临床常用单位
    return round(lavmax_ml, 2)


# 测试函数（可直接运行验证）
if __name__ == "__main__":
    # 模拟左心房掩码
    mask = np.zeros((256, 256))
    mask[50:150, 50:150] = 1  # 模拟左心房区域
    area, length = calculate_la_geometry(mask, pixel_spacing=0.5)
    lavmax = calculate_lavmax_single_plane(area, length)
    print(f"✅ 测试结果：")
    print(f"左心房面积: {area} mm²")
    print(f"长轴长度: {length} mm")
    print(f"估算LAVmax: {lavmax} ml")