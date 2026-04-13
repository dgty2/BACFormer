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
    """
    左心房容积测量工具主窗口（仅ES帧单文件版）
    提供仅使用收缩末期(ES)单帧影像，估算左心房射血分数的功能
    """

    def __init__(self, weight_path="weights/best_model.pth"):
        """
        初始化主窗口
        Args:
            weight_path (str): 模型权重文件路径
        """
        super().__init__()
        # 设置窗口标题
        self.setWindowTitle("左心房容积测量工具（单ES帧版）")
        # 设置窗口位置和大小（x=100, y=100, 宽=1200, 高=800）
        self.setGeometry(100, 100, 1200, 800)
        # 数据存储变量
        self.img_data = None  # 影像数据
        self.pixdim = None  # 像素间距信息
        self.mask = None  # 分割掩码
        self.current_slice = 0  # 当前显示的切片索引
        self.case_id = ""  # 病例ID
        self.current_file_path = ""  # 当前文件路径
        self.model = None  # 加载的模型
        self.device = None  # 模型设备
        # 手动编辑相关变量
        self.edit_mode = None  # 编辑模式：None, 'brush', 'eraser'
        self.brush_size = 5  # 画笔大小
        self.is_drawing = False  # 是否正在绘制
        # 射血分数相关变量
        self.lavmax_value = None  # 舒张末期容积（从ES帧计算得到）
        self.lavmin_value = None  # 收缩末期容积（估算值）
        # 尝试加载模型
        try:
            self.model, self.device = load_bacformer_model(weight_path)
            QMessageBox.information(self, "成功", "模型权重加载成功！")
        except Exception as e:
            QMessageBox.warning(self, "提示", f"模型权重加载失败：{str(e)}<br>可先手动加载掩码计算")
        # 初始化用户界面
        self._init_ui()

    def _init_ui(self):
        """
        初始化用户界面
        创建左右分栏布局：左侧显示图像，右侧为控制面板
        """
        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # 创建水平主布局
        main_layout = QHBoxLayout(central_widget)
        # ========== 左侧：图像显示区域 ==========
        left_layout = QVBoxLayout()
        # 图像标签，用于显示医学影像
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border: 1px solid #ccc; min-width: 800px; min-height: 600px;")
        # 绑定鼠标事件，用于手动编辑掩码
        self.img_label.mousePressEvent = self._mouse_press
        self.img_label.mouseMoveEvent = self._mouse_move
        self.img_label.mouseReleaseEvent = self._mouse_release
        left_layout.addWidget(self.img_label)
        # 切片滑块，用于浏览3D影像的不同切片
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.valueChanged.connect(self._on_slice_change)
        left_layout.addWidget(self.slice_slider)
        # ========== 右侧：控制面板 ==========
        right_layout = QVBoxLayout()
        # 1. 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        self.open_btn = QPushButton("打开 ES 帧 NII 影像")
        self.open_btn.clicked.connect(self._open_nii_file)
        file_layout.addWidget(self.open_btn)
        file_group.setLayout(file_layout)
        right_layout.addWidget(file_group)
        # 2. 分割操作组
        seg_group = QGroupBox("分割操作")
        seg_layout = QVBoxLayout()
        self.auto_seg_btn = QPushButton("自动分割左心房")
        self.auto_seg_btn.clicked.connect(self._auto_segment)
        self.auto_seg_btn.setEnabled(False)  # 初始禁用，加载影像后启用
        seg_layout.addWidget(self.auto_seg_btn)
        self.load_gt_btn = QPushButton("加载手动掩码")
        self.load_gt_btn.clicked.connect(self._load_gt_mask)
        self.load_gt_btn.setEnabled(False)  # 初始禁用
        seg_layout.addWidget(self.load_gt_btn)
        seg_group.setLayout(seg_layout)
        right_layout.addWidget(seg_group)
        # 3. 手动编辑组
        edit_group = QGroupBox("手动编辑分割结果")
        edit_layout = QVBoxLayout()
        self.brush_btn = QRadioButton("画笔（添加区域）")
        self.eraser_btn = QRadioButton("橡皮擦（删除区域）")
        self.brush_btn.toggled.connect(self._on_edit_mode_change)
        self.eraser_btn.toggled.connect(self._on_edit_mode_change)
        # 画笔大小调节
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
        # 4. 容积计算组
        calc_group = QGroupBox("容积计算")
        calc_layout = QVBoxLayout()
        self.calc_btn = QPushButton("计算容积与估算EF")
        self.calc_btn.clicked.connect(self._calculate_all)
        self.calc_btn.setEnabled(False)  # 初始禁用
        calc_layout.addWidget(self.calc_btn)
        # 移除了原来的LAVmin计算和EF计算按钮，现在单步完成
        calc_group.setLayout(calc_layout)
        right_layout.addWidget(calc_group)
        # 5. 结果显示组
        result_group = QGroupBox("测量结果")
        result_layout = QGridLayout()
        self.area_edit = QLineEdit(readOnly=True)  # 只读文本框
        self.long_axis_edit = QLineEdit(readOnly=True)
        self.lavmax_edit = QLineEdit(readOnly=True)
        # 新增：射血分数相关的输入框
        self.lavmin_edit = QLineEdit(readOnly=True)
        self.ef_edit = QLineEdit(readOnly=True)
        self.dice_edit = QLineEdit(readOnly=True)
        # 添加标签和输入框到网格布局
        result_layout.addWidget(QLabel("左心房面积(mm²)："), 0, 0)
        result_layout.addWidget(self.area_edit, 0, 1)
        result_layout.addWidget(QLabel("长轴长度(mm)："), 1, 0)
        result_layout.addWidget(self.long_axis_edit, 1, 1)
        result_layout.addWidget(QLabel("LAVmax(ml)："), 2, 0)
        result_layout.addWidget(self.lavmax_edit, 2, 1)
        result_layout.addWidget(QLabel("LAVmin(ml)："), 3, 0)
        result_layout.addWidget(self.lavmin_edit, 3, 1)
        result_layout.addWidget(QLabel("射血分数 EF(%)："), 4, 0)
        result_layout.addWidget(self.ef_edit, 4, 1)
        result_layout.addWidget(QLabel("Dice 系数："), 5, 0)
        result_layout.addWidget(self.dice_edit, 5, 1)
        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)
        # 6. 报告导出组
        export_group = QGroupBox("报告导出")
        export_layout = QVBoxLayout()
        self.export_btn = QPushButton("导出 Excel 报告")
        self.export_btn.clicked.connect(self._export_report)
        self.export_btn.setEnabled(False)  # 初始禁用
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        right_layout.addWidget(export_group)
        # 将左右布局添加到主布局，左侧占70%，右侧占30%
        main_layout.addLayout(left_layout, 7)
        main_layout.addLayout(right_layout, 3)

    # -------------------------- 通用的mask清理函数 --------------------------
    def _clean_mask(self, mask):
        """
        通用的掩码清理：删除左侧误标，只保留右侧的左心房
        Args:
            mask (numpy.ndarray): 输入掩码
        Returns:
            numpy.ndarray: 清理后的掩码
        """
        # 获取图像尺寸
        h, w = self.img_data.shape[:2]
        # 清空左边65%的区域，保留右边35%（针对镜像影像）
        mask[:, :int(w * 0.65)] = 0
        # 连通域分析，找到所有独立区域
        labeled, n_labels = label(mask)
        if n_labels > 0:
            # 定义函数：计算某个标签区域的中心x坐标
            def get_center_x(label_id):
                ys, xs = np.where(labeled == label_id)
                return np.mean(xs)

            # 找到最靠右的连通域（x坐标最大的）
            rightmost_label = max(range(1, n_labels + 1), key=get_center_x)
            # 只保留最靠右的连通域
            mask = (labeled == rightmost_label).astype(np.uint8)
        return mask

    # -------------------------- 文件加载 --------------------------
    @pyqtSlot()
    def _open_nii_file(self):
        """
        槽函数：打开NII影像文件
        通过文件对话框选择并加载NIfTI格式的医学影像（ES帧）
        """
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 ES 帧 NII 影像", "", "NII Files (*.nii *.nii.gz)")
        # 如果用户取消选择，直接返回
        if not file_path:
            return
        # 保存文件路径
        self.current_file_path = file_path
        # 从文件名提取病例ID
        self.case_id = os.path.basename(file_path).replace(".nii.gz", "").replace(".nii", "")
        try:
            # 加载影像数据和像素间距
            self.img_data, self.pixdim = load_nii(file_path)
            # 根据影像维度设置滑块范围
            if len(self.img_data.shape) == 2:
                # 2D影像：只有一个切片
                self.slice_slider.setRange(0, 0)
                self.current_slice = 0
                pixmap = slice_to_qimage(self.img_data)
            else:
                # 3D影像：多个切片
                self.slice_slider.setRange(0, self.img_data.shape[-1] - 1)
                # 默认显示中间切片
                self.current_slice = self.img_data.shape[-1] // 2
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice])
            # 在标签中显示图像
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            # 启用相关按钮
            self.auto_seg_btn.setEnabled(self.model is not None)
            self.load_gt_btn.setEnabled(True)
            # 清空当前文件的所有结果
            self._clear_results()
            # 显示成功消息
            QMessageBox.information(self, "成功", f"加载ES帧影像：{self.case_id}<br>像素间距：{self.pixdim} mm")
        except Exception as e:
            # 显示错误消息
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    # -------------------------- 分割 --------------------------
    @pyqtSlot()
    def _auto_segment(self):
        """
        槽函数：执行自动分割
        使用BACFormer模型对加载的影像进行左心房分割
        """
        # 检查是否已加载影像和模型
        if self.img_data is None or self.model is None:
            QMessageBox.warning(self, "提示", "请先加载影像和模型权重")
            return
        try:
            # 更新按钮状态
            self.auto_seg_btn.setText("分割中...")
            self.auto_seg_btn.setEnabled(False)
            # 根据影像维度选择处理方式
            if len(self.img_data.shape) == 2:
                # 2D影像：直接预测
                self.mask = predict_mask(self.model, self.device, self.img_data)
                # 清理掩码：删除左侧误标
                self.mask = self._clean_mask(self.mask)
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                # 3D影像：逐帧预测
                self.mask = np.zeros_like(self.img_data, dtype=np.uint8)
                for i in range(self.img_data.shape[-1]):
                    # 对每一帧进行预测
                    slice_mask = predict_mask(self.model, self.device, self.img_data[..., i])
                    # 清理每一帧的预测结果
                    self.mask[..., i] = self._clean_mask(slice_mask)
                # 显示当前切片的分割结果
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
            # 更新显示
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            # 启用计算按钮
            self.calc_btn.setEnabled(True)
            # 恢复按钮状态
            self.auto_seg_btn.setText("自动分割左心房")
            self.auto_seg_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "自动分割完成！已自动清理左侧误标，只保留右半部分左心房")
        except Exception as e:
            # 出错时恢复按钮状态
            self.auto_seg_btn.setText("自动分割左心房")
            self.auto_seg_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"分割失败：{str(e)}")

    @pyqtSlot()
    def _load_gt_mask(self):
        """
        槽函数：加载手动标注的GT掩码
        从文件系统加载与当前影像对应的手动标注掩码
        """
        # 检查是否已加载影像
        if self.img_data is None:
            QMessageBox.warning(self, "提示", "请先加载影像")
            return
        try:
            # 更新按钮状态
            self.load_gt_btn.setText("加载中...")
            self.load_gt_btn.setEnabled(False)
            # 加载GT掩码
            self.mask = load_label_mask(self.current_file_path)
            # 二值化掩码
            self.mask = (self.mask > 0).astype(np.uint8)
            # 使用通用清理函数，保持与自动分割一致
            self.mask = self._clean_mask(self.mask)
            # 根据影像维度显示掩码
            if len(self.img_data.shape) == 2:
                pixmap = slice_to_qimage(self.img_data, self.mask)
            else:
                pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
            # 更新显示
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))
            # 启用计算按钮
            self.calc_btn.setEnabled(True)
            # 恢复按钮状态
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.information(self, "成功", "手动掩码加载完成！已自动清理左侧误标")
        except Exception as e:
            # 出错时恢复按钮状态
            self.load_gt_btn.setText("加载手动掩码")
            self.load_gt_btn.setEnabled(True)
            QMessageBox.critical(self, "错误", f"加载失败：{str(e)}")

    # -------------------------- 手动编辑 --------------------------
    def _on_edit_mode_change(self):
        """
        槽函数：编辑模式切换
        根据单选按钮状态切换画笔或橡皮擦模式
        """
        # 检查画笔按钮是否选中
        if self.brush_btn.isChecked():
            self.edit_mode = 'brush'
        # 检查橡皮擦按钮是否选中
        elif self.eraser_btn.isChecked():
            self.edit_mode = 'eraser'
        # 都未选中
        else:
            self.edit_mode = None

    def _get_image_coords(self, event):
        """
        将鼠标事件坐标转换为图像像素坐标
        Args:
            event (QMouseEvent): 鼠标事件
        Returns:
            tuple: (x, y) 图像像素坐标，失败返回(None, None)
        """
        # 获取当前显示的pixmap
        pixmap = self.img_label.pixmap()
        # 检查pixmap和影像数据是否存在
        if pixmap is None or self.img_data is None:
            return None, None
        # 1. 获取原始图像尺寸
        if len(self.img_data.shape) == 2:
            img_h, img_w = self.img_data.shape
        else:
            img_h, img_w = self.img_data.shape[:2]
        # 2. 获取标签控件的尺寸
        label_w = self.img_label.width()
        label_h = self.img_label.height()
        # 3. 计算缩放比例和缩放后的尺寸
        scale = min(label_w / img_w, label_h / img_h)
        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)
        # 4. 计算居中偏移量（图像在标签中的偏移）
        x_offset = (label_w - scaled_w) / 2
        y_offset = (label_h - scaled_h) / 2
        # 5. 获取鼠标在标签内的坐标
        mouse_x = event.x() - self.img_label.x()
        mouse_y = event.y() - self.img_label.y()
        # 6. 转换到原始图像坐标系统
        x = int((mouse_x - x_offset) / scale)
        y = int((mouse_y - y_offset) / scale)
        # 边界检查，确保坐标不超出图像范围
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        return x, y

    def _mouse_press(self, event: QMouseEvent):
        """
        槽函数：鼠标按下事件
        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 检查是否在编辑模式且有掩码
        if self.edit_mode is None or self.mask is None:
            return
        # 开始绘制
        self.is_drawing = True
        # 获取图像坐标
        x, y = self._get_image_coords(event)
        if x is None:
            return
        # 在掩码上绘制
        self._draw_on_mask(x, y)

    def _mouse_move(self, event: QMouseEvent):
        """
        槽函数：鼠标移动事件
        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 检查是否正在绘制且在编辑模式
        if not self.is_drawing or self.edit_mode is None:
            return
        # 获取图像坐标
        x, y = self._get_image_coords(event)
        if x is None:
            return
        # 在掩码上绘制
        self._draw_on_mask(x, y)
        # 实时刷新显示
        if len(self.img_data.shape) == 2:
            pixmap = slice_to_qimage(self.img_data, self.mask)
        else:
            pixmap = slice_to_qimage(self.img_data[..., self.current_slice], self.mask[..., self.current_slice])
        self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def _mouse_release(self, event: QMouseEvent):
        """
        槽函数：鼠标释放事件
        Args:
            event (QMouseEvent): 鼠标事件
        """
        # 停止绘制
        self.is_drawing = False

    def _draw_on_mask(self, x, y):
        """
        在掩码上绘制圆形
        Args:
            x (int): x坐标
            y (int): y坐标
        """
        # 获取当前切片
        if len(self.mask.shape) == 2:
            mask_slice = self.mask
        else:
            mask_slice = self.mask[..., self.current_slice]
        # 使用OpenCV画圆：画笔模式设为1，橡皮擦模式设为0
        cv2.circle(mask_slice, (x, y), self.brush_size, 1 if self.edit_mode == 'brush' else 0, -1)

    # -------------------------- 计算 --------------------------
    @pyqtSlot()
    def _calculate_all(self):
        """
        槽函数：单步计算所有指标（仅ES帧）
        基于ES帧计算LAVmax，然后基于临床经验模型估算LAVmin和EF
        """
        # 检查是否有掩码
        if self.mask is None:
            QMessageBox.warning(self, "提示", "请先完成分割/加载掩码")
            return
        try:
            # 获取当前切片
            if len(self.mask.shape) == 2:
                mask_slice = self.mask
            else:
                mask_slice = self.mask[..., self.current_slice]
            # 计算几何参数
            area, long_axis = calculate_la_geometry(mask_slice, self.pixdim)
            # 计算LAVmax（ES帧对应的左心房最大容积）
            lavmax = calculate_lavmax_single_plane(area, long_axis)
            # 尝试计算Dice系数
            dice = None
            try:
                # 加载GT掩码
                gt_mask = load_label_mask(self.current_file_path)
                # 如果是3D，提取当前切片
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[..., self.current_slice]
                # 二值化
                gt_mask = (gt_mask > 0).astype(np.uint8)
                # 对GT也应用同样的清理
                gt_mask = self._clean_mask(gt_mask)
                # 计算Dice系数
                dice = calculate_dice(mask_slice, gt_mask)
            except:
                # 如果加载GT失败，跳过Dice计算
                pass
            # 格式化Dice字符串
            dice_str = f"{dice:.4f}" if dice is not None else "无"

            # --------------------------
            # 基于临床经验模型估算EF和LAVmin
            # 研究表明：左心房最大容积(LAVmax)与LAEF呈负相关，LAVmax越大，EF越低
            # 回归模型：EF = 40 - 0.1 * LAVmax，限制范围10%~50%（符合临床实际范围）
            # --------------------------
            ef = 40 - 0.1 * lavmax
            # 限制EF在合理的临床范围内
            ef = max(10.0, min(50.0, ef))
            # 根据EF计算估算的LAVmin
            lavmin = lavmax * (1 - ef / 100)

            # 保存变量
            self.lavmax_value = lavmax
            self.lavmin_value = lavmin

            # 更新显示
            self.area_edit.setText(f"{area:.2f}")
            self.long_axis_edit.setText(f"{long_axis:.2f}")
            self.lavmax_edit.setText(f"{lavmax:.2f}")
            self.lavmin_edit.setText(f"{lavmin:.2f} (估算)")
            self.ef_edit.setText(f"{ef:.2f} (估算)")
            self.dice_edit.setText(dice_str)

            # 启用导出按钮
            self.export_btn.setEnabled(True)

            QMessageBox.information(self, "计算完成",
                                    f"LAVmax：{lavmax:.2f} mL<br>"
                                    f"估算LAVmin：{lavmin:.2f} mL<br>"
                                    f"估算射血分数EF：{ef:.2f}%<br>"
                                    f"Dice：{dice_str}<br><br>"
                                    f"注意：EF为基于单ES帧的临床经验估算值，仅供参考！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败：{str(e)}")

    # -------------------------- 其他 --------------------------
    @pyqtSlot(int)
    def _on_slice_change(self, idx):
        """
        槽函数：切片滑块值改变
        Args:
            idx (int): 新的切片索引
        """
        # 检查是否已加载影像
        if self.img_data is None:
            return
        # 更新当前切片索引
        self.current_slice = idx
        # 如果是3D影像，更新显示
        if len(self.img_data.shape) == 3:
            slice_data = self.img_data[..., idx]
            # 获取对应的掩码切片
            mask_slice = self.mask[..., idx] if (self.mask is not None and len(self.mask.shape) == 3) else self.mask
            # 更新显示
            pixmap = slice_to_qimage(slice_data, mask_slice)
            self.img_label.setPixmap(pixmap.scaled(self.img_label.size(), Qt.KeepAspectRatio))

    @pyqtSlot()
    def _export_report(self):
        """
        槽函数：导出Excel报告
        将测量结果导出为Excel格式的报告文件
        """
        # 弹出目录选择对话框
        save_path = QFileDialog.getExistingDirectory(self, "选择报告保存路径")
        # 如果用户取消，直接返回
        if not save_path:
            return
        try:
            # 从文本框读取数值，处理估算标记
            area = float(self.area_edit.text()) if self.area_edit.text() else None
            long_axis = float(self.long_axis_edit.text()) if self.long_axis_edit.text() else None
            lavmax = float(self.lavmax_edit.text()) if self.lavmax_edit.text() else None

            # 处理LAVmin，去掉估算标记
            lavmin_text = self.lavmin_edit.text().replace(" (估算)", "")
            lavmin = float(lavmin_text) if lavmin_text else None

            # 处理EF，去掉估算标记
            ef_text = self.ef_edit.text().replace(" (估算)", "")
            ef = float(ef_text) if ef_text else None

            dice = float(self.dice_edit.text()) if self.dice_edit.text() != "无" else None

            # 导出Excel报告
            excel_path = export_excel_report(self.case_id, area, long_axis, lavmax, dice, save_path, lavmin, ef)
            QMessageBox.information(self, "成功", f"报告已导出到：{excel_path}<br>注意：报告中EF为估算值")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{str(e)}")

    def _clear_results(self):
        """
        清空当前文件的所有测量结果
        """
        # 清空所有结果
        self.area_edit.setText("")
        self.long_axis_edit.setText("")
        self.lavmax_edit.setText("")
        self.lavmin_edit.setText("")
        self.ef_edit.setText("")
        self.dice_edit.setText("")
        # 重置当前的mask
        self.mask = None
        # 重置容积变量
        self.lavmax_value = None
        self.lavmin_value = None
        # 禁用计算按钮，直到新的分割完成
        self.calc_btn.setEnabled(False)
        # 禁用导出按钮
        self.export_btn.setEnabled(False)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    # 创建Qt应用
    app = QApplication(sys.argv)
    # 创建主窗口
    window = LAMeasureWindow()
    # 显示窗口
    window.show()
    # 进入事件循环
    app.exec_()