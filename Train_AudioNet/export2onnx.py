import os
import argparse
import torch
from model import SnoreClassifier,SnoreClassifierWithAttention
import os
from datetime import datetime

def run():
    print("Train_Audio 开始转换模型")
    print(args)
    if not args.pt.endswith(".pt"):
        print("模型文件格式不正确,pt=%s" % args.pt)
        return

    if not os.path.exists(args.pt):
        print("模型文件不存在,pt=%s" % args.pt)
        return
    pt_model_path = args.pt

    # 加载PyTorch模型
    checkpoint = torch.load(pt_model_path, map_location='cpu')

    # 判断是否使用注意力机制
    if 'attention' in pt_model_path.lower():
        model = SnoreClassifierWithAttention(num_classes=2)
        print('使用带注意力机制的ResNet18')
    else:
        model = SnoreClassifier(num_classes=2)
        print('使用标准ResNet18')

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dummy_input = torch.randn(1, 3, 128, 32)

    # 导出模型到ONNX格式
    input_names = ['input']
    output_names = ['output']

    if os.path.exists(args.onnx):
        os.remove(args.onnx)

    torch.onnx.export(model, dummy_input, args.onnx,
                      verbose=False,
                      input_names=input_names,
                      output_names=output_names,
                      opset_version=10,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                      )

    # torch.onnx.export(model, dummy_input, args.onnx,
    #                   export_params=True,
    #                   verbose=False,
    #                   input_names=input_names,
    #                   output_names=output_names,
    #                   opset_version=10,
    #                   do_constant_folding=True,
    #                   dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    if os.path.exists(args.onnx):
        print("Train_Audio 模型转换成功,转换文件路径：%s" % args.onnx)
    else:
        print("Train_Audio 模型转换失败")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export2onnx')
    parser.add_argument('--pt', default='checkpoints.pt', help='输入模型文件路径，要求pt格式')
    parser.add_argument('--onnx', default="checkpoint.onnx", help='输出模型文件路径，要求onnx格式')
    args = parser.parse_args()

    # args.pt = "checkpoints/best_model.pt"
    # args.onnx = "checkpoints/best_model.onnx"

    run()
