import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import LAMeasureWindow

if __name__ == "__main__":
    if sys.platform == "darwin":
        from PyQt5.QtCore import Qt
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)
    window = LAMeasureWindow(weight_path="weights/best_model.pth")
    window.show()
    sys.exit(app.exec_())