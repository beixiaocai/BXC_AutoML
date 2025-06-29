import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_audio_file, audio_to_melspectrogram
from torchaudio import transforms

class SnoreDataset(Dataset):
    """
    打鼾声音数据集
    """
    def __init__(self, root_dir, transform=None, sample_rate=16000, max_duration=5, is_train=True):
        """
        初始化数据集
        
        Args:
            root_dir: 数据集根目录，应包含snore和non_snore两个子目录
            transform: 数据增强转换
            sample_rate: 采样率
            max_duration: 最大音频长度（秒）
            is_train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.is_train = is_train
        
        # 类别映射
        self.class_map = {'snore': 0, 'non_snore': 1}
        
        # 收集文件路径和标签
        self.files = []
        self.labels = []
        
        for class_name in self.class_map.keys():
            class_dir = os.path.join(root_dir, class_name)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.wav'):
                        self.files.append(os.path.join(class_dir, file_name))
                        self.labels.append(self.class_map[class_name])
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        
        # 加载音频
        waveform = load_audio_file(
            audio_path, 
            sample_rate=self.sample_rate,
            max_duration=self.max_duration
        )
        
        # 转换为梅尔频谱图
        mel_spec = audio_to_melspectrogram(waveform, sample_rate=self.sample_rate)
        
        # 应用数据增强
        if self.transform and self.is_train:
            mel_spec = self.transform(mel_spec)
        
        # 确保形状为 [channel, height, width]
        if len(mel_spec.shape) == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # 复制到3个通道以匹配ResNet输入
        mel_spec = mel_spec.repeat(3, 1, 1)
        
        return mel_spec, label

def get_dataloaders(data_dir, batch_size=32, sample_rate=16000, max_duration=5, num_workers=4):
    """
    创建训练集和验证集的数据加载器
    
    Args:
        data_dir: 数据集根目录
        batch_size: 批量大小
        sample_rate: 采样率
        max_duration: 最大音频长度（秒）
        num_workers: 数据加载线程数
    
    Returns:
        训练集和验证集的数据加载器
    """
    # 数据增强
    train_transform = torch.nn.Sequential(
        # 随机时间掩码
        transforms.TimeMasking(time_mask_param=20),
        # 随机频率掩码
        transforms.FrequencyMasking(freq_mask_param=10)
    )
    
    # 创建数据集
    train_dataset = SnoreDataset(
        os.path.join(data_dir, 'train'),
        transform=train_transform,
        sample_rate=sample_rate,
        max_duration=max_duration,
        is_train=True
    )
    
    val_dataset = SnoreDataset(
        os.path.join(data_dir, 'val'),
        transform=None,
        sample_rate=sample_rate,
        max_duration=max_duration,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader