# 导出 RKNPU 适配模型说明

## Source

​	本仓库基于 https://github.com/ultralytics/ultralytics  仓库的 50497218c25682458ea35b02dcc5d8a364f34591 commit 进行修改,验证.



## 模型差异

在基于不影响输出结果, 不需要重新训练模型的条件下, 有以下改动:

- 修改输出结构, 移除后处理结构. (后处理结果对于量化不友好)

- dfl 结构在 NPU 处理上性能不佳，移至模型外部的后处理阶段，此操作大部分情况下可提升推理性能。


- 模型输出分支新增置信度的总和，用于后处理阶段加速阈值筛选。 


以上移除的操作, 均需要在外部使用CPU进行相应的处理. (对应的后处理代码可以在 **RKNN_Model_Zoo** 中找到)



## 导出onnx模型

在满足 ./requirements.txt 的环境要求后，执行以下语句导出模型

```
# 调整 ./ultralytics/cfg/default.yaml 中 model 文件路径，默认为 yolo11n.pt，若自己训练模型，请调接至对应的路径。支持检测、分割、姿态、旋转框检测模型。
# 如填入 yolo11n.pt 导出检测模型
# 如填入 yolo11n-seg.pt 导出分割模型
# 如填入 yolo11n-pose.pt 导出姿态模型
# 如填入 yolo11n-obb.pt 导出OBB模型

export PYTHONPATH=./
python ./ultralytics/engine/exporter.py

# 执行完毕后，会生成 ONNX 模型. 假如原始模型为 yolo11n.pt，则生成 yolo11n.onnx 模型。
```



## 转RKNN模型、Python demo、C demo

请参考 https://github.com/airockchip/rknn_model_zoo

