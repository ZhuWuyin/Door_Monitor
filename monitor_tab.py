import os
import sys
import cv2
import torch
import numpy as np
import time
from collections import deque
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QFileDialog, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

from model import get_model, preprocess_image
from torchvision import transforms

class CameraThread(QThread):
    """摄像头线程"""
    frame_signal = pyqtSignal(np.ndarray)  # 发送帧信号
    
    def __init__(self, camera_id=0):
        super().__init__()
        self.camera_id = camera_id
        self.running = False
        self.latest_frame = None
        
    def run(self):
        """运行摄像头线程"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return
            
        self.running = True
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("无法读取摄像头帧")
                break
                
            # 保存最新帧并发送信号
            self.latest_frame = frame.copy()
            self.frame_signal.emit(self.latest_frame)
            
            # 控制帧率（30FPS）
            time.sleep(0.03)
            
        cap.release()
    
    def stop(self):
        """停止摄像头线程"""
        self.running = False
        self.wait()

class AIProcessThread(QThread):
    """AI处理线程"""
    result_signal = pyqtSignal(str, float)  # 发送预测结果信号 (类别, 置信度)
    
    def __init__(self, get_latest_frame, model, device):
        super().__init__()
        self.get_latest_frame = get_latest_frame  # 获取最新帧的回调函数
        self.model = model
        self.device = device
        self.running = False
        
    def run(self):
        """运行AI处理线程"""
        try:
            model = self.model

            # 定义类别
            classes = ['close', 'open']
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.running = True
            while self.running:
                # 获取最新帧
                frame = self.get_latest_frame()
                if frame is None:
                    time.sleep(0.03)
                    continue
                    
                try:
                    # 预处理帧
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    processed_frame = transform(rgb_frame)
                    input_tensor = processed_frame.unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        # 获取预测结果
                        predicted_class = classes[predicted.item()]
                        confidence_value = confidence.item()
                        
                        # 发送预测结果信号
                        self.result_signal.emit(predicted_class, confidence_value)
                        
                except Exception as e:
                    print(f"AI处理错误: {str(e)}")
                    continue
                    
                # 控制处理频率（5Hz）
                time.sleep(0.2)
                
        except Exception as e:
            print(f"AI线程错误: {str(e)}")
            
    def stop(self):
        """停止AI处理线程"""
        self.running = False
        self.wait()

class RecordingThread(QThread):
    """录制线程"""
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.running = False
        self.frame_buffer = deque()  # 存储所有帧及其时间戳
        self.last_open_time = None  # 最后一次检测到open的时间
        self.writer = None
        self.current_video_path = None
        
    def run(self):
        """运行录制线程"""
        self.running = True
        while self.running:
            current_time = datetime.now()
            
            # 检查是否需要开始/停止录制
            if self.last_open_time is not None:
                # 如果是首次检测到open或者超过5秒没有新的open
                time_since_open = (current_time - self.last_open_time).total_seconds()
                
                if time_since_open <= 5:
                    # 如果在5秒内再次检测到open，刷新计时
                    if not self.writer:
                        self.start_recording()
                else:
                    # 超过5秒没有新的open，停止录制
                    if self.writer:
                        self.stop_recording()
            
            # 如果正在录制，继续写入帧
            if self.writer:
                try:
                    if len(self.frame_buffer) > 0:
                        frame, timestamp = self.frame_buffer.popleft()
                
                        # 绘制时间水印（左下角）
                        cv2.putText(frame, timestamp.strftime("%Y-%m-%d %H:%M:%S"), (20, frame.shape[0] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # 只录制开门前后5秒内的帧
                        time_diff = (timestamp - self.last_open_time).total_seconds()
                        if -5 <= time_diff <= 10:  # 包含5秒前和最多10秒后的帧
                            self.writer.write(frame)
                except Exception as e:
                    print(f"写入视频错误: {str(e)}")
                    
            # 控制线程频率
            time.sleep(0.025)
            
    def add_frame(self, frame, timestamp):
        """添加帧到缓冲区"""
        self.frame_buffer.append((frame.copy(), timestamp))
        
    def update_open_time(self):
        """更新最后一次检测到open的时间"""
        self.last_open_time = datetime.now()
        
    def start_recording(self):
        """开始录制视频"""
        try:
            # 创建保存目录
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 创建视频文件名
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_video_path = os.path.join(self.save_dir, f"door_open_{timestamp_str}.avi")
            
            # 获取第一帧的尺寸
            if self.frame_buffer:
                first_frame, _ = self.frame_buffer[0]
                height, width = first_frame.shape[:2]
                
                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.writer = cv2.VideoWriter(self.current_video_path, fourcc, 30, (width, height))
                print(f"开始录制视频: {self.current_video_path}")
                
        except Exception as e:
            print(f"开始录制错误: {str(e)}")
            
    def stop_recording(self):
        """停止录制"""
        if self.writer:
            self.writer.release()
            self.writer = None
            print("停止录制")
            
    def stop(self):
        """停止录制线程"""
        self.running = False
        if self.writer:
            self.writer.release()
        self.wait()

class MonitorTab(QWidget):
    """监控标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.camera_thread = None
        self.ai_thread = None
        self.recording_thread = None
        self.latest_frame = None  # 存储最新帧
        self.save_dir = ""
        self.current_status = "close"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 默认设备
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        
        # 摄像头设置区域
        camera_group = QGroupBox("摄像头设置")
        camera_layout = QHBoxLayout()
        
        camera_layout.addWidget(QLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("默认摄像头", 0)
        
        # 检测可用摄像头
        self.detect_cameras()
        
        camera_layout.addWidget(self.camera_combo)
        
        # 添加设备选择下拉菜单
        camera_layout.addWidget(QLabel("选择设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("GPU")
            self.device_combo.setCurrentText("GPU")
        self.device_combo.currentTextChanged.connect(self.device_changed)
        camera_layout.addWidget(self.device_combo)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # 保存设置区域
        save_group = QGroupBox("保存设置")
        save_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("选择保存目录")
        self.save_btn.clicked.connect(self.select_save_dir)
        save_layout.addWidget(self.save_btn)
        
        self.save_label = QLabel("保存目录: 未选择")
        save_layout.addWidget(self.save_label)
        
        save_group.setLayout(save_layout)
        main_layout.addWidget(save_group)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始监控")
        self.start_btn.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止监控")
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(control_layout)
        
        # 视频显示区域
        self.video_label = QLabel("摄像头预览")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label)
        
        # 状态显示区域
        status_layout = QHBoxLayout()
        
        status_layout.addWidget(QLabel("当前状态:"))
        self.status_label = QLabel("未监控")
        self.status_label.setStyleSheet("font-weight: bold; color: gray;")
        status_layout.addWidget(self.status_label)
        
        status_layout.addStretch()
        
        status_layout.addWidget(QLabel("置信度:"))
        self.confidence_label = QLabel("0.00")
        status_layout.addWidget(self.confidence_label)
        
        main_layout.addLayout(status_layout)
        
        self.setLayout(main_layout)

    def device_changed(self, device_text):
        """设备选择改变事件"""
        self.device = "cuda" if device_text == "GPU" and torch.cuda.is_available() else "cpu"
        print(f"监控--选择设备：{self.device}")
        
    def detect_cameras(self):
        """检测可用摄像头"""
        # 清空下拉框
        while self.camera_combo.count() > 1:
            self.camera_combo.removeItem(1)
            
        # 检测摄像头
        for i in range(1, 5):  # 检测前5个摄像头
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"摄像头 {i}", i)
                cap.release()
                
    def select_save_dir(self):
        """选择保存目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if dir_path:
            self.save_dir = dir_path
            self.save_label.setText(f"保存目录: {dir_path}")
            
    def start_monitoring(self):
        """开始监控"""
        # 检查是否选择了保存目录
        if not self.save_dir:
            QMessageBox.warning(self, "错误", "请先选择保存目录")
            return
            
        # 检查是否有训练好的模型
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "找不到训练好的模型，请先训练模型")
            return
                
        # 加载模型
        print(f"AI线程: 使用设备 {self.device}")
        model = get_model(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
                
        # 获取选择的摄像头ID
        camera_id = self.camera_combo.currentData()
        
        # 初始化录制线程
        self.recording_thread = RecordingThread(self.save_dir)
        self.recording_thread.start()
        
        # 启动摄像头线程
        self.camera_thread = CameraThread(camera_id)
        self.camera_thread.frame_signal.connect(self.update_frame)
        self.camera_thread.start()
        
        # 启动AI处理线程
        self.ai_thread = AIProcessThread(self.get_latest_frame, model, device=self.device)
        self.ai_thread.result_signal.connect(self.update_status)
        self.ai_thread.start()
        
        # 更新UI
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.camera_combo.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        # 更新状态
        self.status_label.setText("监控中")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")
        
    def stop_monitoring(self):
        """停止监控"""
        # 停止线程
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
            
        if self.ai_thread:
            self.ai_thread.stop()
            self.ai_thread = None
            
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread = None
            
        # 更新UI
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.camera_combo.setEnabled(True)
        self.save_btn.setEnabled(True)
        
        # 清空视频显示
        self.video_label.clear()
        self.video_label.setText("摄像头预览")
        
        # 更新状态
        self.status_label.setText("未监控")
        self.status_label.setStyleSheet("font-weight: bold; color: gray;")
        self.confidence_label.setText("0.00")
        
    def get_latest_frame(self):
        """获取最新帧的回调函数"""
        return self.latest_frame
        
    def update_frame(self, frame):
        """更新视频帧"""
        # 保存最新帧
        self.latest_frame = frame.copy()
        
        # 添加到录制缓冲区
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.add_frame(frame.copy(), datetime.now())

        # 添加时间水印和AI检测状态水印
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_text = f"Door Status: {self.current_status}"
        
        # 绘制时间水印（左下角）
        cv2.putText(frame, timestamp, (20, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制AI检测状态水印（左上角）
        color = (0, 0, 255) if self.current_status.lower() == "open" else (0, 255, 0)
        cv2.putText(frame, status_text, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 转换帧为QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # 调整大小并显示
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(), 
                                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_status(self, status, confidence):
        """更新状态"""
        self.current_status = status
        
        # 更新状态标签
        if status == "open":
            self.status_label.setText("开门")
            self.status_label.setStyleSheet("font-weight: bold; color: red;")
            # 更新录制线程的最后检测时间
            if self.recording_thread and self.recording_thread.isRunning():
                self.recording_thread.update_open_time()
        else:
            self.status_label.setText("关门")
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
            
        # 更新置信度标签
        self.confidence_label.setText(f"{confidence:.2f}")
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        self.stop_monitoring()
        event.accept()