import os
import sys
import cv2
import torch
import numpy as np
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QFileDialog, QMessageBox, QGroupBox, QListWidget, QProgressBar, QTextEdit)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from model import get_model, preprocess_image
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

class VideoProcessingThread(QThread):
    """视频处理线程"""
    update_signal = pyqtSignal(str)  # 用于更新处理日志
    progress_signal = pyqtSignal(int, int)  # 用于更新进度条 (当前帧, 总帧数)
    finished_signal = pyqtSignal(bool, str)  # 处理完成信号 (是否成功, 消息)
    
    def __init__(self, video_paths, output_dir, model_path='best_model.pth', 
                 frame_interval=5, confidence_threshold=0.7, device=None):
        super().__init__()
        self.video_paths = video_paths
        self.output_dir = output_dir
        self.model_path = model_path
        self.frame_interval = frame_interval
        self.confidence_threshold = confidence_threshold
        self.device = device  # 添加设备参数
        self.running = True
        
    def run(self):
        """运行视频处理"""
        try:
            # 检查模型路径
            if not os.path.exists(self.model_path):
                self.model_path = "best_model.pth"
                if not os.path.exists(self.model_path):
                    self.finished_signal.emit(False, "找不到训练好的模型")
                    return
            
            # 加载模型
            self.update_signal.emit(f"使用设备: {self.device}")
            
            model = get_model(device=self.device)
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))  # 修改此处
            model.eval()
            
            # 创建时间戳文件夹
            timestamp = datetime.now().strftime("%Y.%m.%d_%H-%M")
            result_dir = os.path.join(self.output_dir, f"视频检测结果{timestamp}")
            os.makedirs(result_dir, exist_ok=True)
            
            # 处理每个视频
            for video_idx, video_path in enumerate(self.video_paths):
                if not self.running:
                    self.update_signal.emit("处理被用户中断")
                    break
                
                self.update_signal.emit(f"正在处理视频 {video_idx+1}/{len(self.video_paths)}: {os.path.basename(video_path)}")
                
                # 为每个视频创建单独的文件夹
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(result_dir, video_name)
                os.makedirs(video_output_dir, exist_ok=True)
                
                # 处理视频
                self.process_video(video_path, video_output_dir, model)
            
            if self.running:
                self.finished_signal.emit(True, f"所有视频处理完成，结果保存在: {result_dir}")
            else:
                self.finished_signal.emit(False, "处理被用户中断")
                
        except Exception as e:
            self.update_signal.emit(f"处理出错: {str(e)}")
            self.finished_signal.emit(False, f"处理失败: {str(e)}")
            
    def process_video(self, video_path, output_dir, model):
        """处理单个视频"""
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.update_signal.emit(f"无法打开视频: {video_path}")
            return
        
        # 获取视频属性
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.update_signal.emit(f"视频FPS: {fps}")
        self.update_signal.emit(f"视频分辨率: {frame_width}x{frame_height}")
        self.update_signal.emit(f"总帧数: {total_frames}")
        
        # 创建日志文件
        log_file = os.path.join(output_dir, "detection_log.txt")
        with open(log_file, "w") as f:
            f.write(f"视频检测日志: {video_path}\n")
            f.write(f"检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"视频FPS: {fps}\n")
            f.write(f"视频分辨率: {frame_width}x{frame_height}\n")
            f.write(f"总帧数: {total_frames}\n")
            f.write(f"检测帧间隔: {self.frame_interval}\n")
            f.write(f"置信度阈值: {self.confidence_threshold}\n")
            f.write("-" * 50 + "\n\n")
        
        # 状态变量
        current_status = "关门"  # 初始状态假设为关门
        status_changes = []  # 记录状态变化的时间点
        frame_count = 0
        segment_count = 0
        
        # 当前片段的视频写入器
        segment_writer = None
        segment_frames = []  # 存储当前片段的所有帧
        
        # 处理视频
        while True:
            if not self.running:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存原始帧用于输出
            original_frame = frame.copy()
            
            # 每隔frame_interval帧进行一次检测
            if frame_count % self.frame_interval == 0:
                # 预处理图像
                # 将OpenCV的BGR转换为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # 调整大小并转换为张量
                transform = transforms.Compose([
                    transforms.Resize((300, 300)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(pil_image).unsqueeze(0).to(self.device)
                
                # 进行预测
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    # 获取预测结果
                    predicted_class = "开门" if predicted.item() == 1 else "关门"
                    confidence_value = confidence.item()
                    
                    # 只有当置信度超过阈值时才考虑状态变化
                    if confidence_value >= self.confidence_threshold:
                        # 检查状态是否发生变化
                        if predicted_class != current_status:
                            # 记录状态变化的时间点
                            frame_time = frame_count / fps
                            status_changes.append((frame_count, frame_time, predicted_class))
                            
                            # 记录日志
                            with open(log_file, "a") as f:
                                f.write(f"帧 {frame_count}: 状态变为 {predicted_class}, 时间: {self.format_time(frame_time)}, 置信度: {confidence_value:.4f}\n")
                            
                            self.update_signal.emit(f"帧 {frame_count}: 状态变为 {predicted_class}, 时间: {self.format_time(frame_time)}, 置信度: {confidence_value:.4f}")
                            
                            # 如果状态从close变为open，开始新的片段
                            if predicted_class == "开门" and current_status == "关门":
                                segment_count += 1
                                self.update_signal.emit(f"开始片段 {segment_count}")
                                segment_frames = []  # 清空片段帧列表
                            
                            # 如果状态从open变为close，结束当前片段
                            elif predicted_class == "关门" and current_status == "开门":
                                self.update_signal.emit(f"结束片段 {segment_count}")
                                
                                # 保存当前片段
                                if segment_frames:
                                    segment_output_path = os.path.join(output_dir, f"segment_{segment_count}.mp4")
                                    self.save_segment(segment_frames, segment_output_path, fps, frame_width, frame_height)
                                    self.update_signal.emit(f"保存片段 {segment_count} 到 {segment_output_path}")
                            
                            # 更新当前状态
                            current_status = predicted_class

            # 使用PIL添加中文水印 (替换原来的cv2.putText)
            img_pil = Image.fromarray(cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype("simhei.ttf", 40)
            status_text = f"状态: {current_status} ({confidence_value:.2f})"
            draw.text((10, 10), status_text, font=font, fill=(0, 255, 0))
            
            # 更新原始帧
            original_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            
            # 将当前帧添加到片段中（如果当前状态是open或者正在收集片段）
            if current_status == "开门" or (segment_frames and current_status == "关门"):
                segment_frames.append(original_frame)
            
            frame_count += 1
            
            # 更新进度
            self.progress_signal.emit(frame_count, total_frames)
            
            # 显示进度
            if frame_count % 100 == 0:
                self.update_signal.emit(f"处理进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.2f}%)")
        
        # 如果视频结束时仍在open状态，保存最后一个片段
        if current_status == "开门" and segment_frames:
            segment_output_path = os.path.join(output_dir, f"segment_{segment_count}.mp4")
            self.save_segment(segment_frames, segment_output_path, fps, frame_width, frame_height)
            self.update_signal.emit(f"保存最后片段 {segment_count} 到 {segment_output_path}")
        
        # 释放资源
        cap.release()
        
        # 生成总结报告
        self.generate_summary_report(status_changes, fps, total_frames, output_dir)
        
        self.update_signal.emit("视频处理完成!")
        
    def save_segment(self, frames, output_path, fps, width, height):
        """保存视频片段"""
        if not frames:
            self.update_signal.emit("没有帧可保存")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        self.update_signal.emit(f"片段已保存到 {output_path}, 共 {len(frames)} 帧")
        
    def generate_summary_report(self, status_changes, fps, total_frames, output_dir):
        """生成检测结果摘要报告"""
        summary_path = os.path.join(output_dir, "summary_report.txt")
        
        with open(summary_path, "w") as f:
            f.write("门状态检测摘要报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总帧数: {total_frames}\n")
            f.write(f"视频时长: {self.format_time(total_frames/fps)}\n\n")
            
            f.write("状态变化记录:\n")
            f.write("-" * 50 + "\n")
            
            if not status_changes:
                f.write("未检测到状态变化\n")
            else:
                # 按时间顺序排序
                status_changes.sort(key=lambda x: x[0])
                
                # 记录每个状态变化
                for i, (frame_num, frame_time, status) in enumerate(status_changes):
                    f.write(f"{i+1}. 帧 {frame_num}: 状态变为 {status}, 时间: {self.format_time(frame_time)}\n")
                
                f.write("\n")
                f.write("开门-关门片段:\n")
                f.write("-" * 50 + "\n")
                
                # 分析开门-关门片段
                segments = []
                open_frame = None
                
                for frame_num, frame_time, status in status_changes:
                    if status == "开门":
                        open_frame = (frame_num, frame_time)
                    elif status == "关门" and open_frame is not None:
                        # 找到一个完整的开门-关门片段
                        segments.append((open_frame, (frame_num, frame_time)))
                        open_frame = None
                
                # 如果最后一个状态是open，记录到视频结束
                if open_frame is not None:
                    segments.append((open_frame, (total_frames, total_frames/fps)))
                
                # 记录每个片段
                for i, ((open_frame, open_time), (close_frame, close_time)) in enumerate(segments):
                    duration = close_time - open_time
                    f.write(f"片段 {i+1}:\n")
                    f.write(f"  开始: 帧 {open_frame}, 时间: {self.format_time(open_time)}\n")
                    f.write(f"  结束: 帧 {close_frame}, 时间: {self.format_time(close_time)}\n")
                    f.write(f"  持续时间: {self.format_time(duration)}\n\n")
        
        self.update_signal.emit(f"摘要报告已保存到 {summary_path}")
        
    def format_time(self, seconds):
        """将秒数转换为时:分:秒.毫秒格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}"
        
    def stop(self):
        """停止处理"""
        self.running = False

class VideoTab(QWidget):
    """视频检测标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.video_paths = []
        self.output_dir = ""
        self.processing_thread = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 默认设备
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        
        # 设备选择区域
        device_group = QGroupBox("设备设置")
        device_layout = QHBoxLayout()
        
        device_layout.addWidget(QLabel("选择设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("GPU")
            self.device_combo.setCurrentText("GPU")
        self.device_combo.currentTextChanged.connect(self.device_changed)
        device_layout.addWidget(self.device_combo)
        
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)
        
        # 视频选择区域
        video_group = QGroupBox("视频选择")
        video_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择视频文件")
        self.select_btn.clicked.connect(self.select_videos)
        btn_layout.addWidget(self.select_btn)
        
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_videos)
        btn_layout.addWidget(self.clear_btn)
        
        video_layout.addLayout(btn_layout)
        
        self.video_list = QListWidget()
        video_layout.addWidget(self.video_list)
        
        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group)
        
        # 输出设置
        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout()
        
        self.output_btn = QPushButton("选择输出目录")
        self.output_btn.clicked.connect(self.select_output_dir)
        output_layout.addWidget(self.output_btn)
        
        self.output_label = QLabel("输出目录: 未选择")
        output_layout.addWidget(self.output_label)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # 处理控制
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("开始检测")
        self.start_btn.clicked.connect(self.start_processing)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("停止检测")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(control_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 处理日志
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.setLayout(main_layout)
        
    def device_changed(self, device_text):
        """设备选择改变事件"""
        self.device = "cuda" if device_text == "GPU" and torch.cuda.is_available() else "cpu"
        print(f"视频检测--选择设备: {self.device}")
        
    def select_videos(self):
        """选择视频文件"""
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)", options=options
        )
        
        if not file_paths:
            return
            
        # 添加到列表
        for path in file_paths:
            if path not in self.video_paths:
                self.video_paths.append(path)
                self.video_list.addItem(os.path.basename(path))
                
        self.update_log(f"已选择 {len(self.video_paths)} 个视频文件")
        
    def clear_videos(self):
        """清空视频列表"""
        self.video_paths = []
        self.video_list.clear()
        self.update_log("已清空视频列表")
        
    def select_output_dir(self):
        """选择输出目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_dir = dir_path
            self.output_label.setText(f"输出目录: {dir_path}")
            self.update_log(f"已选择输出目录: {dir_path}")
            
    def start_processing(self):
        """开始处理视频"""
        # 检查是否选择了视频
        if not self.video_paths:
            QMessageBox.warning(self, "错误", "请先选择视频文件")
            return
            
        # 检查是否选择了输出目录
        if not self.output_dir:
            QMessageBox.warning(self, "错误", "请先选择输出目录")
            return
            
        # 检查是否有训练好的模型
        model_path = "best_model.pth"
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "错误", "找不到训练好的模型，请先训练模型")
            return
                
        # 清空日志
        self.log_text.clear()
        
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建并启动处理线程
        self.processing_thread = VideoProcessingThread(
            self.video_paths, self.output_dir, model_path,
            frame_interval=5, confidence_threshold=0.7,
            device=self.device  # 添加设备参数
        )
        self.processing_thread.update_signal.connect(self.update_log)
        self.processing_thread.progress_signal.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.processing_finished)
        self.processing_thread.start()
        
    def stop_processing(self):
        """停止处理视频"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(self, "确认", "确定要停止处理吗？",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.processing_thread.stop()
                self.update_log("正在停止处理...")
                
    def update_log(self, message):
        """更新处理日志"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def update_progress(self, current, total):
        """更新进度条"""
        progress = int(current / total * 100)
        self.progress_bar.setValue(progress)
        
    def processing_finished(self, success, message):
        """处理完成事件"""
        # 启用开始按钮，禁用停止按钮
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 显示消息
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "处理中断", message)