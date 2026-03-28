from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QSlider,
                             QLineEdit, QMessageBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
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

        self.img_data = None
        self.pixdim = None
        self.mask = None
        self.current_slice = 0
        self.case_id = ""
        self.current_file_path = ""
        self.model = None
        self.device = None

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

        left_layout = QVBoxLayout()
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 1px solid #ccc; min-width: 800px; min-height: 600px;")
        left_layout.addWidget(self.img_label)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_change)
        left_layout.addWidget(self.slice_slider)

        right_layout = QVBoxLayout()

        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.open_btn = QPushButton("打开 NII 影像")
        self.open_btn.clicked.connect(self._open_nii_file)
        file_layout.addWidget(self.open_btn)
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)

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

        calc_group = QGroupBox("容积计算")
        calc_layout = QVBoxLayout()
        self.calc_btn = QPushButton("计算 LAVmax")
        self.calc_btn.clicked.connect(self._calculate_lavmax)
        self.calc_btn.setEnabled(False)
        calc_layout.addWidget(self.calc_btn)
        calc_group.setLayout(calc_layout)
        right_layout.addWidget(calc_group)

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
    def _auto_segment(self):
        if self.img_data is None or self.model is None:
            QMessageBox.warning(self, "提示", "请先加载影像和模型权重")
            return
        try:
            self.auto_seg_btn.setText("分割中...")
            self.auto_seg_btn.setEnabled(False)

            if len(self.img_data.shape) == 2:
                self.mask = predict_mask(self.model, self.device, self.img_data)
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                self.mask = np.zeros_like(self.img_data, dtype=np.uint8)
                for i in range(self.img_data.shape[-1]):
                    self.mask[..., i] = predict_mask(self.model, self.device, self.img_data[..., i])
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])

            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            self.calc_btn.setEnabled(True)
            self.auto_seg_btn.setText("自动分割左心房")
            self.auto_seg_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "自动分割完成！")
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
            labeled, n_labels = label(self.mask)
            if n_labels > 0:
                max_label = max(range(1, n_labels + 1), key=lambda x: sum_labels(self.mask == x, labeled))
                self.mask = (labeled == max_label).astype(np.uint8)

            if len(self.img_data.shape) == 2:
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])

            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            self.calc_btn.setEnabled(True)
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "手动掩码加载完成！")
        except Exception as e:
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

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
            QMessageBox.information(self, "成功", f"报告已导出：<br>{excel_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def _clear_results(self):
        self.area_edit.clear()
        self.long_axis_edit.clear()
        self.lavmax_edit.clear()
        self.dice_edit.clear()
        self.export_btn.setEnabled(False)
        self.mask = None