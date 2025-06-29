import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torchaudio import transforms

def load_audio_file(file_path, sample_rate=16000, max_duration=5):
    """
    加载音频文件并进行预处理
    
    Args:
        file_path: 音频文件路径
        sample_rate: 目标采样率
        max_duration: 最大音频长度（秒）
    
    Returns:
        预处理后的音频张量
    """
    # 加载音频
    waveform, sr = torchaudio.load(file_path)
    
    # 如果是双声道，转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样到目标采样率
    if sr != sample_rate:
        resampler = transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    """
    # 裁剪或填充到固定长度
    max_length = max_duration * sample_rate
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    else:
        # 填充
        padding = max_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    """
    return waveform

def audio_to_melspectrogram(waveform, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    """
    将音频转换为梅尔频谱图
    
    Args:
        waveform: 音频波形
        sample_rate: 采样率
        n_mels: 梅尔滤波器组数量
        n_fft: FFT窗口大小
        hop_length: 帧移
    
    Returns:
        梅尔频谱图
    """
    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True
    )(waveform)
    
    # 转换为分贝单位
    mel_spectrogram = transforms.AmplitudeToDB()(mel_spectrogram)
    
    return mel_spectrogram

def plot_spectrogram(spectrogram, title=None, save_path=None):
    """
    绘制频谱图
    
    Args:
        spectrogram: 频谱图张量
        title: 图表标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spectrogram[0].numpy(), aspect='auto', origin='lower', cmap='viridis')
    
    if title:
        ax.set_title(title)
    
    ax.set_ylabel('Mel Bin')
    ax.set_xlabel('Time')
    
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()