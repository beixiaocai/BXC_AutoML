from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.facenet import Facenet
from utils.utils import preprocess_input, resize_image, show_config
import os
import argparse
import torch

def run():
    print("开始转换模型")
    print(args)

    if not args.model_path.endswith(".pth"):
        print("模型文件格式不正确,pt=%s" % args.pt)
        return

    if not os.path.exists(args.model_path):
        print("模型文件不存在,model_path=%s" % args.model_path)
        return

    net = Facenet(backbone="mobilenet", mode="predict").eval()
    net.load_state_dict(torch.load(args.model_path, map_location=args.device), strict=False)
    print('{} model loaded.'.format(args.model_path))

    dummy_input = torch.randn(1, 3, 160, 160, device=args.device)

    # 导出模型到ONNX格式
    input_names = ['input']
    output_names = ['output']

    if os.path.exists(args.onnx):
        os.remove(args.onnx)
    torch.onnx.export(net, dummy_input, args.onnx,
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
        print("模型转换成功,转换文件路径：%s" % args.onnx)
    else:
        print("模型转换失败")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export2onnx')
    parser.add_argument('--model_path', default='checkpoint.pth', help='输入模型文件路径，要求pth格式')
    parser.add_argument('--onnx', default="checkpoint.onnx", help='输出模型文件路径，要求onnx格式')
    parser.add_argument('--device', default="cpu", help='device')

    args = parser.parse_args()
    run()
