import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QComboBox, QLineEdit, QTextEdit, QGroupBox, QGridLayout,
                           QMessageBox, QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

from model import get_model, preprocess_image

class DoorDataset(Dataset):
    """门状态数据集"""
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ['close', 'open']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, self.class_to_idx[class_name]))
                        self.images.append(img_path)
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # 使用预处理函数处理图像
        image = preprocess_image(img_path)
        
        return image, label

class TrainingThread(QThread):
    """训练线程"""
    update_signal = pyqtSignal(str)  # 用于更新训练日志
    progress_signal = pyqtSignal(int, int)  # 用于更新进度条 (当前epoch, 总epochs)
    finished_signal = pyqtSignal(bool, str)  # 训练完成信号 (是否成功, 消息)
    
    def __init__(self, device, num_epochs):
        super().__init__()
        self.device = device
        self.num_epochs = num_epochs
        self.running = True
        
    def run(self):
        """运行训练过程"""
        try:
            # 设置设备
            device = self.device
            self.update_signal.emit(f"使用设备: {device}")
            
            # 创建数据集
            train_dataset = DoorDataset(root_dir='images/train')
            test_dataset = DoorDataset(root_dir='images/test')
            
            self.update_signal.emit(f"训练集大小: {len(train_dataset)}")
            self.update_signal.emit(f"测试集大小: {len(test_dataset)}")
            
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                self.finished_signal.emit(False, "训练集或测试集为空，请先上传照片")
                return
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
            
            # 获取模型
            model = get_model(num_classes=2, device=device)  # 修改此处
            model = model.to(device)  # 确保模型在指定设备上
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            
            # 训练模型
            best_model_wts = None
            best_loss = float('inf')
            best_acc = 0.0
            
            # 记录训练和测试损失
            train_losses = []
            test_losses = []
            test_accuracies = []
            
            for epoch in range(self.num_epochs):
                if not self.running:
                    self.update_signal.emit("训练被用户中断")
                    break
                    
                self.update_signal.emit(f'Epoch {epoch+1}/{self.num_epochs}')
                self.update_signal.emit('-' * 30)
                
                # 更新进度条
                self.progress_signal.emit(epoch + 1, self.num_epochs)
                
                # 训练阶段
                model.train()
                running_loss = 0.0
                
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 随机抽取一部分图片进行通道置零操作
                    # 随机选择要处理的图片数量（10%-30%的batch）
                    num_to_modify = max(1, int(inputs.size(0) * torch.rand(1).item() * 0.2 + 0.1))
                    # 随机选择图片索引
                    indices = torch.randperm(inputs.size(0))[:num_to_modify]
                    
                    for idx in indices:
                        # 随机选择一个通道(R=0, G=1, B=2)
                        channel = torch.randint(0, 3, (1,)).item()
                        # 将选定通道置零
                        inputs[idx, channel, :, :] = 0
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item() * inputs.size(0)
                
                epoch_train_loss = running_loss / len(train_loader.dataset)
                train_losses.append(epoch_train_loss)
                
                # 测试阶段
                model.eval()
                running_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        
                        # 前向传播
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        running_loss += loss.item() * inputs.size(0)
                        
                        # 计算准确率
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                epoch_test_loss = running_loss / len(test_loader.dataset)
                test_losses.append(epoch_test_loss)
                
                # 计算准确率百分比
                accuracy = 100 * correct / total
                test_accuracies.append(accuracy)
                
                self.update_signal.emit(f'Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
                
                # 如果测试损失更低，保存模型
                if epoch_test_loss < best_loss and best_acc <= accuracy:
                    best_loss = epoch_test_loss
                    best_acc = accuracy
                    best_model_wts = model.state_dict().copy()
                    torch.save(best_model_wts, 'best_model.pth')
                    self.update_signal.emit(f'保存新的最佳模型，测试损失: {best_loss:.4f}, 测试准确率: {best_acc:.2f}%')
            
            # 训练完成
            if self.running:
                # 绘制损失曲线
                self.plot_losses(train_losses, test_losses, test_accuracies)
                
                # 加载最佳模型权重
                model.load_state_dict(best_model_wts)
                
                self.finished_signal.emit(True, f"训练完成！最佳测试准确率: {best_acc:.2f}%")
            else:
                self.finished_signal.emit(False, "训练被用户中断")
                
        except Exception as e:
            self.update_signal.emit(f"训练出错: {str(e)}")
            self.finished_signal.emit(False, f"训练失败: {str(e)}")
            
    def plot_losses(self, train_losses, test_losses, test_accuracies):
        """绘制损失曲线和准确率曲线"""
        try:
            # 创建图表目录
            os.makedirs(os.path.join('charts'), exist_ok=True)
            
            # 绘制损失曲线
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Testing Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('charts', 'loss_curve.png'))
            plt.close()
            
            # 绘制准确率曲线
            plt.figure(figsize=(10, 5))
            plt.plot(test_accuracies, label='Test Accuracy', color='green')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Testing Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('charts', 'accuracy_curve.png'))
            plt.close()
            
            self.update_signal.emit("损失曲线和准确率曲线已保存")
        except Exception as e:
            self.update_signal.emit(f"绘制曲线出错: {str(e)}")
            
    def stop(self):
        """停止训练"""
        self.running = False

class TrainTab(QWidget):
    """训练标签页"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.training_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        main_layout = QVBoxLayout()
        
        # 训练设置区域
        settings_group = QGroupBox("训练设置")
        settings_layout = QGridLayout()
        
        # 设备选择
        settings_layout.addWidget(QLabel("设备:"), 0, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItem("CPU")
        if torch.cuda.is_available():
            self.device_combo.addItem("GPU")
        self.device_combo.currentTextChanged.connect(self.device_changed)
        settings_layout.addWidget(self.device_combo, 0, 1)
        
        # 训练轮数
        settings_layout.addWidget(QLabel("训练轮数:"), 1, 0)
        self.epochs_input = QLineEdit("50")
        settings_layout.addWidget(self.epochs_input, 1, 1)
        
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)
        
        # 训练控制按钮
        btn_layout = QHBoxLayout()
        
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(self.train_btn)
        
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        main_layout.addLayout(btn_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # 训练日志
        log_group = QGroupBox("训练日志")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # 结果显示
        results_group = QGroupBox("训练结果")
        results_layout = QHBoxLayout()
        
        self.loss_chart = QLabel("损失曲线将在训练后显示")
        self.loss_chart.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.loss_chart)
        
        self.acc_chart = QLabel("准确率曲线将在训练后显示")
        self.acc_chart.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.acc_chart)
        
        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)
        
        self.setLayout(main_layout)
        
    def device_changed(self, device_text):
        """设备选择改变事件"""
        if device_text == "GPU" and not torch.cuda.is_available():
            QMessageBox.warning(self, "警告", "没有检测到GPU，将使用CPU进行训练")
            self.device_combo.setCurrentText("CPU")
            
    def start_training(self):
        """开始训练"""
        # 获取训练参数
        device_text = self.device_combo.currentText()
        device = torch.device("cuda" if device_text == "GPU" and torch.cuda.is_available() else "cpu")
        
        try:
            num_epochs = int(self.epochs_input.text())
            if num_epochs <= 0:
                raise ValueError("训练轮数必须大于0")
        except ValueError as e:
            QMessageBox.warning(self, "错误", f"无效的训练轮数: {str(e)}")
            return
            
        # 清空日志
        self.log_text.clear()
        
        # 禁用开始按钮，启用停止按钮
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建并启动训练线程
        self.training_thread = TrainingThread(device, num_epochs)
        self.training_thread.update_signal.connect(self.update_log)
        self.training_thread.progress_signal.connect(self.update_progress)
        self.training_thread.finished_signal.connect(self.training_finished)
        self.training_thread.start()
        
    def stop_training(self):
        """停止训练"""
        if self.training_thread and self.training_thread.isRunning():
            reply = QMessageBox.question(self, "确认", "确定要停止训练吗？",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.training_thread.stop()
                self.update_log("正在停止训练...")
                
    def update_log(self, message):
        """更新训练日志"""
        self.log_text.append(message)
        # 滚动到底部
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        
    def update_progress(self, current, total):
        """更新进度条"""
        progress = int(current / total * 100)
        self.progress_bar.setValue(progress)
        
    def training_finished(self, success, message):
        """训练完成事件"""
        # 启用开始按钮，禁用停止按钮
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 显示消息
        if success:
            QMessageBox.information(self, "完成", message)
            
            # 显示损失曲线和准确率曲线
            loss_curve_path = os.path.join('charts', 'loss_curve.png')
            acc_curve_path = os.path.join('charts', 'accuracy_curve.png')
            
            if os.path.exists(loss_curve_path):
                loss_pixmap = QPixmap(loss_curve_path)
                self.loss_chart.setPixmap(loss_pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                
            if os.path.exists(acc_curve_path):
                acc_pixmap = QPixmap(acc_curve_path)
                self.acc_chart.setPixmap(acc_pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            QMessageBox.warning(self, "训练中断", message)