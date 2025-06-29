import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SnoreClassifier, SnoreClassifierWithAttention
from utils import load_audio_file, audio_to_melspectrogram, plot_spectrogram

def predict_audio(model, audio_path, device, sample_rate=16000, max_duration=5):
    """
    预测单个音频文件的类别
    
    Args:
        model: 加载的模型
        audio_path: 音频文件路径
        device: 设备
        sample_rate: 采样率
        max_duration: 最大音频长度（秒）
    
    Returns:
        预测类别和概率
    """
    # 加载音频
    waveform = load_audio_file(
        audio_path, 
        sample_rate=sample_rate,
        max_duration=max_duration
    )
    
    # 转换为梅尔频谱图
    mel_spec = audio_to_melspectrogram(waveform, sample_rate=sample_rate)
    
    # 确保形状为 [channel, height, width]
    if len(mel_spec.shape) == 2:
        mel_spec = mel_spec.unsqueeze(0)
    
    # 复制到3个通道以匹配ResNet输入
    mel_spec = mel_spec.repeat(3, 1, 1)
    
    # 添加批次维度
    mel_spec = mel_spec.unsqueeze(0)
    
    # 可视化频谱图
    plot_path = os.path.splitext(audio_path)[0] + '_spectrogram.png'
    plot_spectrogram(mel_spec[0], title='梅尔频谱图', save_path=plot_path)
    print(f'频谱图已保存至 {plot_path}')
    
    # 预测
    model.eval()
    with torch.no_grad():
        mel_spec = mel_spec.to(device)
        print("mel_spec.shape:",mel_spec.shape)

        outputs = model(mel_spec)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # 获取预测类别和概率
        prob, predicted = torch.max(probabilities, 1)
        
    return predicted.item(), prob.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--audio_path', type=str, required=True, help='音频文件路径')
    parser.add_argument('--use_attention', action='store_true', help='使用注意力机制')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--max_duration', type=float, default=5.0, help='最大音频长度（秒）')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建模型
    if args.use_attention:
        model = SnoreClassifierWithAttention(num_classes=2)
        print('使用带注意力机制的ResNet18')
    else:
        model = SnoreClassifier(num_classes=2)
        print('使用标准ResNet18')

    # 加载模型权重
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f'已加载模型: {args.model_path}')
    print(f'模型训练了 {checkpoint["epoch"]} 个epoch')
    print(f'验证准确率: {checkpoint["val_acc"]:.2f}%')

    # 预测
    class_id, probability = predict_audio(
        model,
        args.audio_path,
        device,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )

    # 输出结果
    class_names = ['打鼾', '非打鼾']
    print(f'\n预测结果:')
    print(f'音频文件: {args.audio_path}')
    print(f'预测类别: {class_names[class_id]}')
    print(f'置信度: {probability:.4f} ({probability * 100:.2f}%)')
