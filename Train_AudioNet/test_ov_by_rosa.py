import os
import argparse
import numpy as np
import wave
import scipy.signal
from openvino.runtime import Core
import librosa

def load_audio_file(file_path, sample_rate=16000, max_duration=5):
    """
    使用Python标准库加载音频文件

    Args:
        file_path: 音频文件路径
        sample_rate: 目标采样率
        max_duration: 最大音频长度（秒）

    Returns:
        音频波形数据
    """
    # 使用wave模块读取WAV文件
    with wave.open(file_path, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()

        # 读取音频数据
        frames = wav_file.readframes(n_frames)

    # 将字节数据转换为NumPy数组
    dtype = np.int16 if sample_width == 2 else np.int8
    audio_data = np.frombuffer(frames, dtype=dtype)

    # 如果是双声道，转换为单声道
    if n_channels > 1:
        audio_data = audio_data.reshape(-1, n_channels)
        audio_data = np.mean(audio_data, axis=1)

    # 重采样到目标采样率
    if framerate != sample_rate:
        audio_data = scipy.signal.resample(
            audio_data,
            int(len(audio_data) * sample_rate / framerate)
        )

    """
    # 裁剪或填充到固定长度
    max_length = max_duration * sample_rate
    if len(audio_data) > max_length:
        audio_data = audio_data[:max_length]
    else:
        # 填充
        padding = max_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')
        
    """

    return audio_data


def audio_to_melspectrogram(waveform, sample_rate=16000, n_mels=128, n_fft=1024, hop_length=512):
    """
    优化后的梅尔频谱转换函数，确保帧数与torchaudio一致
    """
    # 计算STFT
    _, _, stft = scipy.signal.stft(
        waveform,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        padded=False,
    )

    # 创建梅尔滤波器组
    mel_basis = librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=0,
        fmax=sample_rate / 2,
        norm='slaney',
        htk=False
    )

    # 应用梅尔滤波器组
    mel_spec = np.dot(mel_basis, np.abs(stft) ** 2)

    # 转换为分贝单位，使用与torchaudio相同的参数
    mel_spec = 20 * np.log10(np.maximum(1e-10, mel_spec))

    # 确保输出形状与torchaudio一致 [n_mels, time]
    mel_spec = mel_spec.astype(np.float32)

    return mel_spec

def predict_audio_openvino(model, audio_path, sample_rate=16000, max_duration=5):
    """
    使用OpenVINO预测单个音频文件的类别

    Args:
        model: 加载的OpenVINO模型
        audio_path: 音频文件路径
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
        # mel_spec = mel_spec.unsqueeze(0)
        mel_spec = np.expand_dims(mel_spec, axis=0)

    # 复制到3个通道以匹配ResNet输入
    mel_spec = np.repeat(mel_spec, 3, axis=0)  # 复制3次以匹配RGB格式

    # 可视化频谱图
    # plot_path = os.path.splitext(audio_path)[0] + '_spectrogram.png'
    # plot_spectrogram(mel_spec[0], title='梅尔频谱图', save_path=plot_path)
    # print(f'频谱图已保存至 {plot_path}')

    mel_spec = np.expand_dims(mel_spec, axis=0)

    # 将PyTorch张量转换为NumPy数组
    input_data = mel_spec.astype(np.float32)

    # 使用OpenVINO进行推理
    # 获取模型的输入名称
    input_name = model.inputs[0].get_any_name()

    # 执行推理
    result = model({input_name: input_data})

    # 获取输出结果
    output_name = model.outputs[0].get_any_name()
    output = result[output_name]

    # 计算softmax概率
    exp_output = np.exp(output)
    probabilities = exp_output / np.sum(exp_output, axis=1, keepdims=True)

    # 获取预测类别和概率
    class_id = np.argmax(probabilities, axis=1)[0]
    probability = probabilities[0, class_id]

    return class_id, probability


def main():
    parser = argparse.ArgumentParser(description='test_ov_by_rosa')
    parser.add_argument('--model_path', type=str, help='OpenVINO模型文件路径(.xml)')
    parser.add_argument('--audio_path', type=str, help='音频文件路径')
    parser.add_argument('--use_attention', action='store_true', help='使用注意力机制')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--max_duration', type=float, default=5.0, help='最大音频长度（秒）')
    parser.add_argument('--device', type=str, default='CPU', help='推理设备 (CPU, GPU, MYRIAD等)')
    args = parser.parse_args()

    # 初始化OpenVINO Core
    core = Core()

    # 加载模型
    model = core.read_model(args.model_path)
    compiled_model = core.compile_model(model, args.device)

    print(f'已加载OpenVINO模型: {args.model_path}')
    print(f'使用设备: {args.device}')

    # 打印模型信息
    print(f'模型输入: {model.inputs[0]}')
    print(f'模型输出: {model.outputs[0]}')

    # 预测
    class_id, probability = predict_audio_openvino(
        compiled_model,
        args.audio_path,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration
    )

    # 输出结果
    class_names = ['打鼾', '非打鼾']
    print(f'\n预测结果:')
    print(f'音频文件: {args.audio_path}')
    print(f'预测类别: {class_names[class_id]}')
    print(f'置信度: {probability:.4f} ({probability * 100:.2f}%)')


if __name__ == '__main__':
    main()