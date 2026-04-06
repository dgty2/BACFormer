import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import LAMeasureWindow

if __name__ == "__main__":
    # 检测是否为macOS系统（darwin）
    if sys.platform == "darwin":
        from PyQt5.QtCore import Qt
        # 启用高DPI缩放，适配Retina显示屏
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        # 使用高分辨率像素图，提高图像清晰度
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    # 创建Qt应用程序实例，传入命令行参数
    app = QApplication(sys.argv)
    
    # 创建主窗口实例，指定模型权重路径
    window = LAMeasureWindow(weight_path="weights/best_model.pth")
    
    # 显示主窗口
    window.show()
    
    # 进入Qt事件循环，程序在此等待用户交互
    # exec_()返回退出码，通过sys.exit()传递给操作系统
    sys.exit(app.exec_())