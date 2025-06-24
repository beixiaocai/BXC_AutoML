import cv2
import numpy as np

from rknn.api import RKNN
import os

if __name__ == '__main__':

    MODEL_DIR = "models"
    onnx_filename = "ResNet_model.onnx"
    platform = 'rk3588'
    do_quant = True # i8,u8,fp
    width = 224
    height = 224

    onnx_model_path = MODEL_DIR + "/" + onnx_filename
    rknn_model_path = '{MODEL_DIR}/{onnx_filename}-to-{platform}-{width}x{height}.rknn'.format(
        MODEL_DIR=MODEL_DIR,
        onnx_filename=onnx_filename,
        platform=platform,
        width=width,
        height=height)


    # Create RKNN object
    rknn = RKNN(verbose=False)
    DATASET = './dataset.txt'
    rknn.config(mean_values=[[0.485, 0.456, 0.406]], std_values=[[
                255*0.229, 255*0.224, 255*0.225]], target_platform=platform)
    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=onnx_model_path, inputs=['input'], input_size_list=[[1, 3, height, width]])
    # ret = rknn.load_onnx(model=onnx_model_path)

    if ret != 0:
        print('load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET)
    if ret != 0:
        print('build model failed.')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model: {}'.format(rknn_model_path))
    ret = rknn.export_rknn(rknn_model_path)
    if ret != 0:
        print('Export rknn model failed.')
        exit(ret)
    print('done')

    rknn.release()

