import os
import argparse
import numpy as np
from openvino.runtime import Core
from utils import load_audio_file, audio_to_melspectrogram, plot_spectrogram

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
        mel_spec = mel_spec.unsqueeze(0)
    
    # 复制到3个通道以匹配ResNet输入
    mel_spec = mel_spec.repeat(3, 1, 1)
    
    # 添加批次维度
    mel_spec = mel_spec.unsqueeze(0)
    
    # 可视化频谱图
    plot_path = os.path.splitext(audio_path)[0] + '_spectrogram.png'
    plot_spectrogram(mel_spec[0], title='梅尔频谱图', save_path=plot_path)
    print(f'频谱图已保存至 {plot_path}')
    
    # 将PyTorch张量转换为NumPy数组
    input_data = mel_spec.numpy()
    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test_ov_by_torchaudio')
    parser.add_argument('--model_path', type=str, help='OpenVINO模型文件路径(.xml)')
    parser.add_argument('--audio_path', type=str, help='音频文件路径')
    parser.add_argument('--use_attention', action='store_true', help='使用注意力机制')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--max_duration', type=float, default=1.0, help='最大音频长度（秒）')
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