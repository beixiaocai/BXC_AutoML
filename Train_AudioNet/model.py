import torch
import torch.nn as nn
import torchvision.models as models

class SnoreClassifier(nn.Module):
    """
    基于ResNet18的打鼾声音分类器
    """
    def __init__(self, num_classes=2, pretrained=True):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
        """
        super(SnoreClassifier, self).__init__()
        
        # 加载预训练的ResNet18，但移除最后的全连接层
        resnet = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 分类层
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
        
        Returns:
            分类结果
        """
        # 特征提取
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        # 分类
        x = self.fc(x)
        
        return x

class SnoreClassifierWithAttention(nn.Module):
    """
    带有注意力机制的ResNet18打鼾声音分类器
    """
    def __init__(self, num_classes=2, pretrained=True):
        """
        初始化模型
        
        Args:
            num_classes: 类别数量
            pretrained: 是否使用预训练权重
        """
        super(SnoreClassifierWithAttention, self).__init__()
        
        # 加载预训练的ResNet18，但移除最后的全连接层
        resnet = models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # 注意力层
        self.attention = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, channels, height, width]
        
        Returns:
            分类结果
        """
        # ResNet特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [batch_size, 512, h, w]
        
        # 应用注意力机制
        attention_weights = self.attention(x)  # [batch_size, 1, h, w]
        x = x * attention_weights  # 加权特征
        
        # 全局平均池化
        x = self.avgpool(x)  # [batch_size, 512, 1, 1]
        x = torch.flatten(x, 1)  # [batch_size, 512]
        
        # 分类
        x = self.fc(x)
        
        return x