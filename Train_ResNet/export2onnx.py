import os
import argparse
import torch
from utils import CreateLogger

logger = CreateLogger(logDir="log", prefix="export2onnx", is_show_console=True)


def run():
    logger.info("Train_ResNet 开始转换模型")
    logger.info(args)
    if not args.pt.endswith(".pt"):
        logger.error("模型文件格式不正确,pt=%s" % args.pt)
        return

    if not os.path.exists(args.pt):
        logger.error("模型文件不存在,pt=%s" % args.pt)
        return

    model = torch.load(args.pt)
    dummy_input = torch.randn(1, 3, 224, 224, device=args.device)

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
        logger.info("Train_ResNet 模型转换成功,转换文件路径：%s" % args.onnx)
    else:
        logger.error("Train_ResNet 模型转换失败")


if __name__ == '__main__':
    # 通过本项目训练的.pt模型转.onnx模型
    parser = argparse.ArgumentParser(description='Train_ResNet export2onnx')
    parser.add_argument('--pt', default='checkpoint.pt', help='输入模型文件路径，要求pt格式')
    parser.add_argument('--onnx', default="checkpoint.onnx", help='输出模型文件路径，要求onnx格式')
    parser.add_argument('--device', default="cpu", help='device')

    args = parser.parse_args()

    run()
