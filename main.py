import os
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PyQt5.QtCore import Qt
import torch

# 导入各个标签页
from photo_tab import PhotoTab
from train_tab import TrainTab
from video_tab import VideoTab
from monitor_tab import MonitorTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("门状态检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 检查必要的目录结构
        self.check_directories()
        
        # 初始化模型
        self.model = None
        self.model_loaded = False
        
        # 检查GPU可用性
        self.has_gpu = torch.cuda.is_available()
        
        # 创建主界面
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        # 创建主标签页
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        
        # 添加标签页
        self.photo_tab = PhotoTab(self)
        self.train_tab = TrainTab(self)
        self.video_tab = VideoTab(self)
        self.monitor_tab = MonitorTab(self)
        
        self.tab_widget.addTab(self.photo_tab, "照片")
        self.tab_widget.addTab(self.train_tab, "训练")
        self.tab_widget.addTab(self.video_tab, "视频检测")
        self.tab_widget.addTab(self.monitor_tab, "监控")
        
    def check_directories(self):
        """检查并创建必要的目录结构"""
        # 创建图像目录
        for category in ["train", "test"]:
            for status in ["open", "close"]:
                os.makedirs(f"images/{category}/{status}", exist_ok=True)
                
        # 创建APP目录下的charts目录
        os.makedirs("charts", exist_ok=True)
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        reply = QMessageBox.question(self, "确认退出", "确定要退出程序吗？",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # 确保所有线程都已停止
            if hasattr(self.monitor_tab, 'stop_monitoring'):
                self.monitor_tab.stop_monitoring()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())