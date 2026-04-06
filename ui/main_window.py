from PyQt5.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider,
                             QLineEdit, QMessageBox, QGroupBox, QGridLayout,
                             QRadioButton, QSpinBox)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QMouseEvent
import os
import numpy as np
from scipy.ndimage import label, sum_labels
import cv2
from core.model import load_bacformer_model, predict_mask
from core.metrics import calculate_la_geometry, calculate_lavmax_single_plane, calculate_dice
from core.utils import load_nii, load_label_mask, slice_to_qimage, export_excel_report


class LAMeasureWindow(QMainWindow):
    def __init__(self, weight_path="weights/best_model.pth"):
        super().__init__()
        self.setWindowTitle("左心房容积测量工具")
        self.setGeometry(100, 100, 1200, 800)
        # 数据存储（仅保留四腔心）
        self.img_data = None
        self.pixdim = None
        self.mask = None
        self.current_slice = 0
        self.case_id = ""
        self.current_file_path = ""
        self.model = None
        self.device = None
        # 手动编辑相关（坐标已修复）
        self.edit_mode = None  # None, 'brush', 'eraser'
        self.brush_size = 5
        self.is_drawing = False
        # 加载模型
        try:
            self.model, self.device = load_bacformer_model(weight_path)
            QMessageBox.information(self, "成功", "模型权重加载成功！")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"模型权重加载失败：{str(e)}<br>可先手动加载掩码计算")
        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        # 左侧：图像显示
        left_layout = QVBoxLayout()
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 1px solid #ccc; min-width: 800px; min-height: 600px;")
        # 鼠标事件，用于手动编辑
        self.img_label.mousePressEvent = self._mouse_press
        self.img_label.mouseMoveEvent = self._mouse_move
        self.img_label.mouseReleaseEvent = self._mouse_release
        left_layout.addWidget(self.img_label)
        # 切片滑块
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_change)
        left_layout.addWidget(self.slice_slider)
        # 右侧：控制面板
        right_layout = QVBoxLayout()
        # 1. 文件操作
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.open_btn = QPushButton("打开 NII 影像")
        self.open_btn.clicked.connect(self._open_nii_file)
        file_layout.addWidget(self.open_btn)
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)
        # 2. 分割操作
        seg_group = QGroupBox("分割操作")
        seg_layout = QVBoxLayout()
        self.auto_seg_btn = QPushButton("自动分割左心房")
        self.auto_seg_btn.clicked.connect(self._auto_segment)
        self.auto_seg_btn.setEnabled(False)
        seg_layout.addWidget(self.auto_seg_btn)
        self.load_gt_btn = QPushButton("加载手动掩码")
        self.load_gt_btn.clicked.connect(self._load_gt_mask)
        self.load_gt_btn.setEnabled(False)
        seg_layout.addWidget(self.load_gt_btn)
        seg_group.setLayout(seg_layout)
        right_layout.addWidget(seg_group)
        # 3. 手动编辑
        edit_group = QGroupBox("手动编辑分割结果")
        edit_layout = QVBoxLayout()
        self.brush_btn = QRadioButton("画笔（添加区域）")
        self.eraser_btn = QRadioButton("橡皮擦（删除区域）")
        self.brush_btn.toggled.connect(self._on_edit_mode_change)
        self.eraser_btn.toggled.connect(self._on_edit_mode_change)
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("画笔大小:"))
        self.brush_size_spin = QSpinBox()
        self.brush_size_spin.setRange(1, 20)
        self.brush_size_spin.setValue(5)
        self.brush_size_spin.valueChanged.connect(lambda v: setattr(self, 'brush_size', v))
        size_layout.addWidget(self.brush_size_spin)
        edit_layout.addWidget(self.brush_btn)
        edit_layout.addWidget(self.eraser_btn)
        edit_layout.addLayout(size_layout)
        edit_group.setLayout(edit_layout)
        right_layout.addWidget(edit_group)
        # 4. 容积计算
        calc_group = QGroupBox("容积计算")
        calc_layout = QVBoxLayout()
        self.calc_btn = QPushButton("计算 LAVmax")
        self.calc_btn.clicked.connect(self._calculate_lavmax)
        self.calc_btn.setEnabled(False)
        calc_layout.addWidget(self.calc_btn)
        calc_group.setLayout(calc_layout)
        right_layout.addWidget(calc_group)
        # 5. 结果显示
        result_group = QGroupBox("测量结果")
        result_layout = QGridLayout()
        self.area_edit = QLineEdit(readOnly=True)
        self.long_axis_edit = QLineEdit(readOnly=True)
        self.lavmax_edit = QLineEdit(readOnly=True)
        self.dice_edit = QLineEdit(readOnly=True)
        result_layout.addWidget(QLabel("左心房面积(mm²)："), 0, 0)
        result_layout.addWidget(self.area_edit, 0, 1)
        result_layout.addWidget(QLabel("长轴长度(mm)："), 1, 0)
        result_layout.addWidget(self.long_axis_edit, 1, 1)
        result_layout.addWidget(QLabel("LAVmax(ml)："), 2, 0)
        result_layout.addWidget(self.lavmax_edit, 2, 1)
        result_layout.addWidget(QLabel("Dice 系数："), 3, 0)
        result_layout.addWidget(self.dice_edit, 3, 1)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        # 6. 导出
        export_group = QGroupBox("报告导出")
        export_layout = QVBoxLayout()
        self.export_btn = QPushButton("导出 Excel 报告")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)
        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)

    # -------------------------- 通用的mask清理函数 --------------------------
    def _clean_mask(self, mask):
        """通用的mask清理：删掉左边的误标，只保留右边的左心房"""
        h, w = self.img_data.shape[:2]
        # 清空左边65%的标注，保留右边35%
        mask[:, :int(w * 0.65)] = 0
        # 清理小噪点，只留最靠右的连通域
        labeled, n_labels = label(mask)
        if n_labels > 0:
            def get_center_x(label_id):
                ys, xs = np.where(labeled == label_id)
                return np.mean(xs)

            rightmost_label = max(range(1, n_labels + 1), key=get_center_x)
            mask = (labeled == rightmost_label).astype(np.uint8)
        return mask

    # -------------------------- 文件加载 --------------------------
    @pyqtSlot()
    def _open_nii_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 NII 影像", "", "NII Files (*.nii *.nii.gz)")
        if not file_path:
            return
        self.current_file_path = file_path
        self.case_id = os.path.basename(file_path).replace(".nii.gz", "").replace(".nii", "")
        try:
            self.img_data, self.pixdim = load_nii(file_path)
            if len(self.img_data.shape) == 2:
                self.slice_slider.setRange(0, 0)
                self.current_slice = 0
                pixmap = slice_to_qimage(self.img_data)
            else:
                self.slice_slider.setRange(0, self.img_data.shape[-1] - 1)
                self.current_slice = self.img_data.shape[-1] // 2
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice])
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            self.auto_seg_btn.setEnabled(self.model is not None)
            self.load_gt_btn.setEnabled(True)
            self._clear_results()
            QMessageBox.information(self, "成功", f"加载影像：{self.case_id}<br>像素间距：{self.pixdim} mm")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    # -------------------------- 分割 --------------------------
    @pyqtSlot()
    def _auto_segment(self):
        if self.img_data is None or self.model is None:
            QMessageBox.warning(self, "提示", "请先加载影像和模型权重")
            return
        try:
            self.auto_seg_btn.setText("分割中...")
            self.auto_seg_btn.setEnabled(False)
            if len(self.img_data.shape) == 2:
                # 先预测
                self.mask = predict_mask(self.model, self.device, self.img_data)
                # 然后做通用的清理！删掉左边的误标！
                self.mask = self._clean_mask(self.mask)
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                self.mask = np.zeros_like(self.img_data, dtype=np.uint8)
                for i in range(self.img_data.shape[-1]):
                    # 逐帧预测
                    slice_mask = predict_mask(self.model, self.device, self.img_data[..., i])
                    # 逐帧清理
                    self.mask[..., i] = self._clean_mask(slice_mask)
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            self.calc_btn.setEnabled(True)
            self.auto_seg_btn.setText("自动分割左心房")
            self.auto_seg_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "自动分割完成！已自动清理左侧误标，只保留右半部分左心房")
        except Exception as e:
            self.auto_seg_btn.setText("自动分割左心房")
            self.auto_seg_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"分割失败：{str(e)}")

    @pyqtSlot()
    def _load_gt_mask(self):
        if self.img_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像")
            return
        try:
            self.load_gt_btn.setText("加载中...")
            self.load_gt_btn.setEnabled(False)
            self.mask = load_label_mask(self.current_file_path)
            self.mask = (self.mask > 0).astype(np.uint8)

            # 用通用的清理函数，和自动分割保持完全一致
            self.mask = self._clean_mask(self.mask)

            if len(self.img_data.shape) == 2:
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            self.calc_btn.setEnabled(True)
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "手动掩码加载完成！已自动清理左侧误标")
        except Exception as e:
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    # -------------------------- 手动编辑 --------------------------
    def _on_edit_mode_change(self):
        if self.brush_btn.isChecked():
            self.edit_mode = 'brush'
        elif self.eraser_btn.isChecked():
            self.edit_mode = 'eraser'
        else:
            self.edit_mode = None

    def _get_image_coords(self, event):
        # 修复后的坐标转换：计算居中偏移，把鼠标坐标转成图像像素坐标
        pixmap = self.img_label.pixmap()
        if pixmap is None or self.img_data is None:
            return None, None
        # 1. 原始图像尺寸
        if len(self.img_data.shape) == 2:
            img_h, img_w = self.img_data.shape
        else:
            img_h, img_w = self.img_data.shape[:2]
        # 2. 标签的尺寸
        label_w = self.img_label.width()
        label_h = self.img_label.height()
        # 3. 缩放后的图像尺寸
        scale = min(label_w / img_w, label_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        # 4. 计算居中的偏移
        x_offset = (label_w - scaled_w) / 2
        y_offset = (label_h - scaled_h) / 2
        # 5. 鼠标在标签内的坐标，减去偏移
        mouse_x = event.x() - self.img_label.x()
        mouse_y = event.y() - self.img_label.y()
        # 6. 转换到原始图像坐标
        x = int((mouse_x - x_offset) / scale)
        y = int((mouse_y - y_offset) / scale)
        # 边界检查，防止越界
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        return x, y

    def _mouse_press(self, event: QMouseEvent):
        if self.edit_mode is None or self.mask is None:
            return
        self.is_drawing = True
        x, y = self._get_image_coords(event)
        if x is None:
            return
        self._draw_on_mask(x, y)

    def _mouse_move(self, event: QMouseEvent):
        if not self.is_drawing or self.edit_mode is None:
            return
        x, y = self._get_image_coords(event)
        if x is None:
            return
        self._draw_on_mask(x, y)
        # 实时刷新显示
        if len(self.img_data.shape) == 2:
            pixmap = slice_to_qimage(self.img_data, self.mask)
        else:
            pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
        self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def _mouse_release(self, event: QMouseEvent):
        self.is_drawing = False

    def _draw_on_mask(self, x, y):
        if len(self.mask.shape) == 2:
            mask_slice = self.mask
        else:
            mask_slice = self.mask[..., self.current_slice]
        # 画圆
        cv2.circle(mask_slice, (x, y), self.brush_size, 1 if self.edit_mode == 'brush' else 0, -1)

    # -------------------------- 计算 --------------------------
    @pyqtSlot()
    def _calculate_lavmax(self):
        if self.mask is None:
            QMessageBox.warning(self, "提示", "请先完成分割/加载掩码")
            return
        try:
            if len(self.mask.shape) == 2:
                mask_slice = self.mask
            else:
                mask_slice = self.mask[..., self.current_slice]
            area, long_axis = calculate_la_geometry(mask_slice, self.pixdim)
            lavmax = calculate_lavmax_single_plane(area, long_axis)
            dice = None
            try:
                gt_mask = load_label_mask(self.current_file_path)
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[..., self.current_slice]
                gt_mask = (gt_mask > 0).astype(np.uint8)
                # 对GT也用同样的清理，保证两边一致
                gt_mask = self._clean_mask(gt_mask)
                # 现在计算的Dice就是纯左心房的相似度了
                dice = calculate_dice(mask_slice, gt_mask)
            except:
                pass
            dice_str = f"{dice:.4f}" if dice is not None else "无"
            self.area_edit.setText(f"{area:.2f}")
            self.long_axis_edit.setText(f"{long_axis:.2f}")
            self.lavmax_edit.setText(f"{lavmax:.2f}")
            self.dice_edit.setText(dice_str)
            self.export_btn.setEnabled(True)
            QMessageBox.information(self, "计算完成",
                                    f"LAVmax：{lavmax:.2f} mL<br>Dice：{dice_str}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败：{str(e)}")

    # -------------------------- 其他 --------------------------
    @pyqtSlot(int)
    def _on_slice_change(self, idx):
        if self.img_data is None:
            return
        self.current_slice = idx
        if len(self.img_data.shape) == 3:
            slice_data = self.img_data[..., idx]
            mask_slice = self.mask[..., idx] if (self.mask is not None and len(self.mask.shape) == 3) else self.mask
            pixmap = slice_to_qimage(slice_data, mask_slice)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))

    @pyqtSlot()
    def _export_report(self):
        save_path = QFileDialog.getExistingDirectory(self, "选择报告保存路径")
        if not save_path:
            return
        try:
            area = float(self.area_edit.text())
            long_axis = float(self.long_axis_edit.text())
            lavmax = float(self.lavmax_edit.text())
            dice = float(self.dice_edit.text()) if self.dice_edit.text() != "无" else None
            excel_path = export_excel_report(self.case_id, area, long_axis, lavmax, dice, save_path)
            QMessageBox.information(self, "成功", f"报告已导出到：{excel_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def _clear_results(self):
        self.area_edit.setText("")
        self.long_axis_edit.setText("")
        self.lavmax_edit.setText("")
        self.dice_edit.setText("")
        self.export_btn.setEnabled(False)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = LAMeasureWindow()
    window.show()
    sys.exit(app.exec_())