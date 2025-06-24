import cv2
import numpy as np

from rknn.api import RKNN
import os

if __name__ == '__main__':

    MODEL_DIR = "models"
    onnx_filename = "rk-yolov8n.onnx"
    platform = 'rk3588'
    do_quant = True # i8,u8,fp
    width = 640
    height = 640

    onnx_model_path = MODEL_DIR + "/" + onnx_filename
    rknn_model_path = '{MODEL_DIR}/{onnx_filename}-to-{platform}-{width}x{height}.rknn'.format(
        MODEL_DIR=MODEL_DIR,
        onnx_filename=onnx_filename.split(".")[0],
        platform=platform,
        width=width,
        height=height)


    # Create RKNN object
    rknn = RKNN(verbose=False)
    DATASET = './dataset.txt'
    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform=platform)
    # Load model
    print('--> Loading model')
    #ret = rknn.load_onnx(model=onnx_model_path, inputs=['input'], input_size_list=[[1, 3, height, width]])
    ret = rknn.load_onnx(model=onnx_model_path)

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

