import os
import sys
import shutil
from datetime import datetime
import torch
import numpy as np
import cv2
from PIL import Image, ImageQt, ImageDraw, ImageFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
                            QComboBox, QLineEdit, QScrollArea, QGroupBox, QGridLayout,
                            QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QMenu, QAction,
                            QInputDialog, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QMimeData
from PyQt5.QtGui import QPixmap, QImage, QDrag, QCursor, QPainter, QPen, QFont

from model import get_model, preprocess_image

class PhotoItem(QLabel):
    """自定义照片项，支持拖放和右键菜单"""
    def __init__(self, img_path, category, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.category = category  # "train" 或 "test"
        self.door_status = os.path.basename(os.path.dirname(img_path))  # "open" 或 "close"
        
        # 加载图像并设置为标签
        pixmap = QPixmap(img_path)
        scaled_pixmap = pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled_pixmap)
        self.setAlignment(Qt.AlignCenter)
        
        # 设置工具提示
        self.setToolTip(f"路径: {img_path}\n类别: {self.door_status}")
        
        # 启用鼠标跟踪
        self.setMouseTracking(True)
        
    def mousePressEvent(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 开始拖动
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.img_path)
            drag.setMimeData(mime_data)
            drag.exec_()
        elif event.button() == Qt.RightButton:
            # 显示右键菜单
            self.show_context_menu(event.pos())
            
    def show_context_menu(self, pos):
        """显示右键菜单"""
        context_menu = QMenu(self)
        delete_action = QAction("删除", self)
        delete_action.triggered.connect(self.delete_photo)
        context_menu.addAction(delete_action)
        
        # 在鼠标位置显示菜单
        context_menu.exec_(self.mapToGlobal(pos))
        
    def delete_photo(self):
        """删除照片"""
        reply = QMessageBox.question(self, "确认删除", 
                                    f"确定要删除这张照片吗？\n{self.img_path}",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                os.remove(self.img_path)
                # 通过父组件链查找PhotoTab实例
                parent = self.parent()
                while parent is not None:
                    if hasattr(parent, 'reload_photos'):
                        parent.reload_photos()
                        break
                    parent = parent.parent()
                QMessageBox.information(self, "成功", "照片已成功删除")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"删除照片失败: {str(e)}")

class PhotoDropArea(QScrollArea):
    """可拖放照片的区域"""
    def __init__(self, category, door_status, parent=None):
        super().__init__(parent)
        self.category = category  # "train" 或 "test"
        self.door_status = door_status  # "open" 或 "close"
        
        # 设置接受拖放
        self.setAcceptDrops(True)
        
        # 创建内容窗口部件
        self.content_widget = QWidget()
        self.setWidget(self.content_widget)
        self.setWidgetResizable(True)
        
        # 创建网格布局
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.content_widget.setLayout(self.grid_layout)
        
    def dragEnterEvent(self, event):
        """拖动进入事件"""
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
            
    def dropEvent(self, event):
        """放下事件"""
        if event.mimeData().hasUrls():
            # 从外部拖入文件
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.confirm_copy_photo(file_path)
        elif event.mimeData().hasText():
            # 从其他照片区域拖入
            file_path = event.mimeData().text()
            self.confirm_copy_photo(file_path)
            
    def confirm_copy_photo(self, file_path):
        """确认是否复制照片"""
        reply = QMessageBox.question(self, "确认", 
                                    f"确定要将照片拷贝到{self.category}中的{self.door_status}集合中吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # 确保目标目录存在
                target_dir = os.path.join("images", self.category, self.door_status)
                os.makedirs(target_dir, exist_ok=True)
                
                # 复制文件
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_dir, file_name)
                
                # 如果目标文件已存在，添加时间戳
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(file_name)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_name = f"{name}_{timestamp}{ext}"
                    target_path = os.path.join(target_dir, file_name)
                
                shutil.copy2(file_path, target_path)
                
                # 重新加载照片
                self.parent().parent().reload_photos()
                QMessageBox.information(self, "成功", f"照片已成功复制到 {target_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"复制照片失败: {str(e)}")

@staticmethod
def generate_gradcam(self, model, img_path):
    """生成GradCAM并叠加在原图上"""
    # 预处理图像并移动到指定设备
    image = preprocess_image(img_path)
    input_tensor = image.unsqueeze(0).to(self.device)

    # 获取原始图像
    try:
        # 使用文件流读取中文路径图片
        with open(img_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            original_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        raise ValueError(f"读取图片失败: {str(e)}")

    # 注册钩子
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    # 获取目标层
    target_layer = model.model.blocks[-1]

    # 注册钩子
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_backward_hook(backward_hook)

    # 前向传播
    model.eval()
    output = model(input_tensor)

    # 获取预测类别
    self.pred_class = output.argmax(dim=1).item()

    # 反向传播
    model.zero_grad()
    class_loss = output[0, self.pred_class]
    class_loss.backward()

    # 移除钩子
    forward_handle.remove()
    backward_handle.remove()

    # 计算权重
    weights = torch.mean(gradients[0], dim=[2, 3], keepdim=True)

    # 生成CAM
    cam = torch.sum(weights * activations[0], dim=1).squeeze()
    cam = cam.detach().cpu().numpy()

    # 归一化
    cam = np.maximum(cam, 0)  # ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # 调整大小以匹配原始图像
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # 创建热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加热力图和原始图像
    alpha = 0.5  # 透明度
    superimposed_img = heatmap * alpha + original_image * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    # 将numpy数组转换为PIL图像以便添加文字
    pil_img = Image.fromarray(superimposed_img)
    draw = ImageDraw.Draw(pil_img)
    
    # 设置字体
    try:
        # 根据图片高度动态计算字体大小
        font_size = max(12, int(original_image.shape[0] * 0.08))
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 获取预测结果和正确答案
    pred_status = "open" if self.pred_class == 1 else "close"
    true_status = "open" if "open" in os.path.basename(os.path.dirname(img_path)) else "close"
    
    # 添加文字标注
    text_y_offset = int(original_image.shape[0] * 0.03)
    draw.text((10, text_y_offset), f"pred: {pred_status}", font=font, fill=(255, 255, 255))
    draw.text((10, text_y_offset * 4), f"answer: {true_status}", font=font, fill=(255, 255, 255))
    
    # 转换回numpy数组
    return np.array(pil_img)

class GradCAMThread(QThread):
    """后台线程用于生成GradCAM分析图"""
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, img_path, device, model_path):
        super().__init__()
        self.img_path = img_path
        self.device = device
        self.model_path = model_path
        self.pred_class = None  # 存储预测类别

    def run(self):
        try:
            # 加载模型
            model = get_model(device=self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

            # 生成GradCAM
            gradcam_img = self.generate_gradcam(model, self.img_path)
            self.finished.emit(gradcam_img)
        except Exception as e:
            self.error.emit(str(e))

    def generate_gradcam(self, model, img_path):
        return generate_gradcam(self, model, img_path)

class PhotoViewer(QMainWindow):
    """照片查看器窗口"""
    def __init__(self, img_path, parent=None, device=None):
        super().__init__(parent)
        self.img_path = img_path
        self.device = device
        self.setWindowTitle(f"照片查看 - {os.path.basename(img_path)}")
        self.setGeometry(200, 200, 800, 400)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建水平布局
        layout = QHBoxLayout(central_widget)

        # 左侧：原始图像
        self.original_view = QGraphicsView()
        self.original_scene = QGraphicsScene()
        self.original_view.setScene(self.original_scene)

        # 右侧：GradCAM叠加图像
        self.gradcam_view = QGraphicsView()
        self.gradcam_scene = QGraphicsScene()
        self.gradcam_view.setScene(self.gradcam_scene)

        # 加载状态标签
        self.loading_label = QLabel("正在生成分析图...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.gradcam_scene.addWidget(self.loading_label)

        # 添加到布局，并设置相同的宽度比例
        layout.addWidget(self.original_view, stretch=1)
        layout.addWidget(self.gradcam_view, stretch=1)

        # 加载图像
        self.load_images()
        
    def showEvent(self, event):
        """窗口显示事件，确保窗口完全显示后再调整图片"""
        super().showEvent(event)
        self.original_view.fitInView(self.original_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        if hasattr(self, 'gradcam_scene') and self.gradcam_scene.items():
            self.gradcam_view.fitInView(self.gradcam_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def load_images(self):
        """加载原始图像和GradCAM叠加图像"""
        try:
            # 加载原始图像
            original_pixmap = QPixmap(self.img_path)
            self.original_scene.clear()
            self.original_scene.addPixmap(original_pixmap)
            self.original_scene.setSceneRect(self.original_scene.itemsBoundingRect())
            self.original_view.fitInView(self.original_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

            # 检查是否有训练好的模型
            model_path = "best_model.pth"

            if os.path.exists(model_path):
                # 显示加载状态
                self.gradcam_scene.clear()
                self.loading_label = QLabel("正在生成分析图...")
                self.loading_label.setAlignment(Qt.AlignCenter)
                self.gradcam_scene.addWidget(self.loading_label)

                # 启动后台线程生成GradCAM
                self.gradcam_thread = GradCAMThread(self.img_path, self.device, model_path)
                self.gradcam_thread.finished.connect(self.update_gradcam)
                self.gradcam_thread.error.connect(self.handle_gradcam_error)
                self.gradcam_thread.start()
            else:
                # 如果没有模型，显示提示
                self.gradcam_scene.clear()
                text_item = self.gradcam_scene.addText("没有找到训练好的模型")
                text_item.setDefaultTextColor(Qt.red)
                self.gradcam_scene.setSceneRect(self.gradcam_scene.itemsBoundingRect())

        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载图像失败: {str(e)}")

    def update_gradcam(self, gradcam_img):
        """更新GradCAM分析图"""
        # 显示GradCAM叠加图像
        height, width, channel = gradcam_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(gradcam_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        gradcam_pixmap = QPixmap.fromImage(q_img)

        self.gradcam_scene.clear()
        self.gradcam_scene.addPixmap(gradcam_pixmap)
        self.gradcam_scene.setSceneRect(self.gradcam_scene.itemsBoundingRect())
        self.gradcam_view.fitInView(self.gradcam_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def handle_gradcam_error(self, error_msg):
        """处理GradCAM生成错误"""
        self.gradcam_scene.clear()
        text_item = self.gradcam_scene.addText(f"生成分析图失败: {error_msg}")
        text_item.setDefaultTextColor(Qt.red)
        self.gradcam_scene.setSceneRect(self.gradcam_scene.itemsBoundingRect())
        
    def resizeEvent(self, event):
        """窗口大小改变事件"""
        self.original_view.fitInView(self.original_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.gradcam_view.fitInView(self.gradcam_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        super().resizeEvent(event)

class PhotoTab(QWidget):
    """照片标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 默认设备
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        
        # 设备选择区域
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("选择设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("GPU")
            self.device_combo.setCurrentText("GPU")
        self.device_combo.currentTextChanged.connect(self.device_changed)
        device_layout.addWidget(self.device_combo)
        main_layout.addLayout(device_layout)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        # 上传照片按钮
        self.upload_btn = QPushButton("上传照片")
        self.upload_btn.clicked.connect(self.upload_photos)
        btn_layout.addWidget(self.upload_btn)
        
        # 导出分析图按钮
        self.export_btn = QPushButton("导出分析图")
        self.export_btn.clicked.connect(self.export_analysis)
        btn_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 照片展示区域
        photos_layout = QHBoxLayout()
        
        # 训练集
        train_group = QGroupBox("训练集")
        train_layout = QVBoxLayout()
        
        # 训练集-开门
        train_open_label = QLabel("开门照片")
        train_layout.addWidget(train_open_label)
        self.train_open_area = PhotoDropArea("train", "open", self)
        train_layout.addWidget(self.train_open_area)
        
        # 训练集-关门
        train_close_label = QLabel("关门照片")
        train_layout.addWidget(train_close_label)
        self.train_close_area = PhotoDropArea("train", "close", self)
        train_layout.addWidget(self.train_close_area)
        
        train_group.setLayout(train_layout)
        photos_layout.addWidget(train_group)
        
        # 测试集
        test_group = QGroupBox("测试集")
        test_layout = QVBoxLayout()
        
        # 测试集-开门
        test_open_label = QLabel("开门照片")
        test_layout.addWidget(test_open_label)
        self.test_open_area = PhotoDropArea("test", "open", self)
        test_layout.addWidget(self.test_open_area)
        
        # 测试集-关门
        test_close_label = QLabel("关门照片")
        test_layout.addWidget(test_close_label)
        self.test_close_area = PhotoDropArea("test", "close", self)
        test_layout.addWidget(self.test_close_area)
        
        test_group.setLayout(test_layout)
        photos_layout.addWidget(test_group)
        
        main_layout.addLayout(photos_layout)
        
        self.setLayout(main_layout)
        
        # 加载照片
        self.reload_photos()
        
    def reload_photos(self):
        """重新加载所有照片"""
        # 清空所有照片区域
        self.clear_photo_area(self.train_open_area)
        self.clear_photo_area(self.train_close_area)
        self.clear_photo_area(self.test_open_area)
        self.clear_photo_area(self.test_close_area)
        
        # 加载训练集-开门照片
        self.load_photos("images/train/open", self.train_open_area)
        
        # 加载训练集-关门照片
        self.load_photos("images/train/close", self.train_close_area)
        
        # 加载测试集-开门照片
        self.load_photos("images/test/open", self.test_open_area)
        
        # 加载测试集-关门照片
        self.load_photos("images/test/close", self.test_close_area)
        
    def clear_photo_area(self, area):
        """清空照片区域"""
        while area.grid_layout.count():
            item = area.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
                
    def load_photos(self, dir_path, area):
        """加载指定目录的照片"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            return
            
        # 获取目录中的所有图片
        image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # 添加到网格布局
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(dir_path, img_file)
            photo_item = PhotoItem(img_path, area.category, area)
            photo_item.mousePressEvent = lambda event, path=img_path, item=photo_item: self.photo_clicked(event, path, item)
            
            row = i // 3
            col = i % 3
            area.grid_layout.addWidget(photo_item, row, col)
            
    def photo_clicked(self, event, img_path, photo_item):
        """照片点击事件"""
        if event.button() == Qt.LeftButton:
            # 左键点击，打开照片查看器
            viewer = PhotoViewer(img_path, self, self.device)
            viewer.show()
        else:
            # 其他按键，调用原始事件处理
            PhotoItem.mousePressEvent(photo_item, event)
            
    def device_changed(self, device_text):
        """设备选择改变事件"""
        self.device = "cuda" if device_text == "GPU" and torch.cuda.is_available() else "cpu"
        print(f"图片展示--选择设备：{self.device}")
        
    def upload_photos(self):
        """上传照片"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择照片", "", "图像文件 (*.jpg *.jpeg *.png)", options=options
        )
        
        if not file_paths:
            return
            
        # 询问用户选择目标位置
        categories = ["训练集-开门", "训练集-关门", "测试集-开门", "测试集-关门"]
        category, ok = QInputDialog.getItem(
            self, "选择目标位置", "请选择照片保存位置:", categories, 0, False
        )
        
        if not ok:
            return
            
        # 确定目标目录
        if category == "训练集-开门":
            target_dir = "images/train/open"
        elif category == "训练集-关门":
            target_dir = "images/train/close"
        elif category == "测试集-开门":
            target_dir = "images/test/open"
        else:  # 测试集-关门
            target_dir = "images/test/close"
            
        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)
        
        # 复制文件
        copied_count = 0
        for file_path in file_paths:
            try:
                file_name = os.path.basename(file_path)
                target_path = os.path.join(target_dir, file_name)
                
                # 如果目标文件已存在，添加时间戳
                if os.path.exists(target_path):
                    name, ext = os.path.splitext(file_name)
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    file_name = f"{name}_{timestamp}{ext}"
                    target_path = os.path.join(target_dir, file_name)
                
                shutil.copy2(file_path, target_path)
                copied_count += 1
            except Exception as e:
                QMessageBox.warning(self, "错误", f"复制文件失败: {str(e)}")
                
        # 重新加载照片
        self.reload_photos()
        
        QMessageBox.information(self, "成功", f"已成功上传 {copied_count} 张照片")
        
    def export_analysis(self):
        """导出分析图"""
        # 检查是否有训练好的模型
        model_path = "best_model.pth"
            
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "没有找到训练好的模型，无法生成分析图")
            return
            
        # 选择导出目录
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not export_dir:
            return
        
        export_dir = os.path.join(export_dir, "analysis")
        try:
            os.makedirs(export_dir, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "错误", f"创建目录失败: {str(e)}")
            return
        
        # 加载模型
        try:
            model = get_model(device=self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()

        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载模型失败: {str(e)}")
            return
            
        # 获取所有照片路径
        all_photos = []
        for category in ["train", "test"]:
            for status in ["open", "close"]:
                dir_path = f"images/{category}/{status}"
                if os.path.exists(dir_path):
                    for img_file in os.listdir(dir_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            all_photos.append(os.path.join(dir_path, img_file))
        
        if not all_photos:
            QMessageBox.warning(self, "错误", "没有找到任何照片")
            return
            
        # 创建进度对话框
        progress = QProgressDialog("正在导出分析图...", "取消", 0, len(all_photos), self)
        progress.setWindowTitle("导出进度")
        progress.setWindowModality(Qt.WindowModal)
        
        # 导出分析图
        exported_count = 0
        for i, img_path in enumerate(all_photos):
            progress.setValue(i)
            if progress.wasCanceled():
                break
                
            try:
                # 使用静态方法同步生成
                gradcam_img = generate_gradcam(self, model, img_path)
                
                # 保存图像
                file_name = os.path.basename(img_path)
                name, ext = os.path.splitext(file_name)
                export_path = os.path.join(export_dir, f"{name}_analysis{ext}")
                
                # 将numpy数组转换为PIL图像并保存
                Image.fromarray(gradcam_img).save(export_path)
                
                exported_count += 1
            except Exception as e:
                print(f"导出分析图失败: {str(e)}")
                
        progress.setValue(len(all_photos))
        
        QMessageBox.information(self, "成功", f"已成功导出 {exported_count} 张分析图到 {export_dir}")