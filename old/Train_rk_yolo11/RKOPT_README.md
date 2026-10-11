# RKNN optimization for exporting model

## Source
Base on https://github.com/ultralytics/ultralytics with commit id as 50497218c25682458ea35b02dcc5d8a364f34591




## What different
With inference result values unchanged, the following optimizations were applied:
- Change output node, remove post-process from the model. (post-process block in model is unfriendly for quantization)
- Remove dfl structure at the end of the model. (which slowdown the inference speed on NPU device)
- Add a score-sum output branch to speedup post-process.

All the removed operation will be done on CPU. (the CPU post-process could be found in **RKNN_Model_Zoo**)




## Export ONNX model

After meeting the environment requirements specified in "./requirements.txt," execute the following command to export the model (support detect/segment/pose/obb model):

```
# Adjust the model file path in "./ultralytics/cfg/default.yaml" (default is yolo11n.pt). If you trained your own model, please provide the corresponding path. 
# For example, filled with yolo11n.pt for detection model.
# Filling with yolo11n-seg.pt for segmentation model.
# Filling with yolo11n-pose.pt for pose model.
# Filling with yolo11n-obb.pt for obb model.

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# Upon completion, the ".onnx" model will be generated. If the original model is "yolo11n.pt," the generated model will be "yolo11n.onnx"
```



## Convert to RKNN model, Python demo, C demo

Please refer to https://github.com/airockchip/rknn_model_zoo.