import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from torchvision import transforms

class DoorDetectionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DoorDetectionModel, self).__init__()
        
        # 创建一个EfficientNet-B0模型，但修改第一层以接受4个通道
        # 首先加载预训练的模型
        self.model = timm.create_model('efficientnet_b2', pretrained=True)
        
        # 获取原始第一层的权重
        original_conv = self.model.conv_stem
        original_weights = original_conv.weight.data
        
        # 创建新的第一层，接受4个通道而不是3个
        self.model.conv_stem = nn.Conv2d(
            4, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False if original_conv.bias is None else True
        )
        
        # 初始化新层的权重：前3个通道使用预训练权重，第4个通道使用随机初始化
        with torch.no_grad():
            self.model.conv_stem.weight[:, :3, :, :] = original_weights
            # 第4个通道（边缘检测）使用预训练权重的平均值进行初始化
            self.model.conv_stem.weight[:, 3:4, :, :] = torch.mean(original_weights, dim=1, keepdim=True)
        
        # 修改分类器以匹配我们的类别数
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        # x的形状应该是[batch_size, 3, height, width]
        batch_size, channels, height, width = x.shape
        
        # 将输入转换为numpy数组以进行边缘检测
        edge_maps = []
        for i in range(batch_size):
            # 将图像从PyTorch格式转换为OpenCV格式
            img = x[i].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            
            # 将图像转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 应用Canny边缘检测
            edges = cv2.Canny(np.uint8(gray * 255), 100, 200)
            
            # 归一化边缘图
            edges = edges.astype(np.float32) / 255.0
            
            # 将边缘图添加到列表中
            edge_maps.append(torch.from_numpy(edges).unsqueeze(0))  # [1, H, W]
        
        # 将边缘图堆叠成一个批次
        edge_batch = torch.stack(edge_maps).to(x.device)  # [batch_size, 1, H, W]
        
        # 将原始图像和边缘图连接起来
        x_with_edges = torch.cat([x, edge_batch], dim=1)  # [batch_size, 4, H, W]
        
        # 将连接后的输入传递给模型
        return self.model(x_with_edges)

def get_model(num_classes=2, device=None):
    """获取模型并移动到指定设备"""
    model = DoorDetectionModel(num_classes)
    if device is not None:
        model = model.to(device)
    return model

def preprocess_image(image_path, image_size=300):
    """
    预处理图像：转换为RGB，调整大小，归一化
    支持中文路径
    """
    # 定义转换
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        # 使用PIL直接读取图像（支持中文路径）
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # 应用转换
        image = transform(image)
        
        return image
    except Exception as e:
        print(f"使用PIL读取图像失败: {str(e)}")
        print("尝试使用OpenCV读取...")
        
        # 如果PIL失败，尝试使用OpenCV的替代方法
        import numpy as np
        
        # 使用numpy读取文件（支持中文路径）
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
            
        # 转换为RGB（OpenCV默认是BGR）
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 将图像转换为PIL图像
        image = transforms.ToPILImage()(image)
        
        # 应用转换
        image = transform(image)
        
        return image