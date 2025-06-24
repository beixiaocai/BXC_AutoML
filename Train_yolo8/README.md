### Train_yolo8
* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_AutoML
* github开源地址：https://github.com/beixiaocai/BXC_AutoML


### 安装Python（Linux建议Python3.8，Windows建议Python3.10）
* [python-官网下载地址](https://www.python.org/getit/)
* [python-夸克网盘下载地址](https://pan.quark.cn/s/72df133d1343)


### 推荐使用Python虚拟环境
* 在使用python开发项目时，推荐使用python的虚拟环境，因为同一台电脑上很可能会安装多个python项目，而不同的python项目可能会使用不同的依赖库，为了避免依赖库不同而导致的冲突，强烈建议使用python虚拟环境
* 关于如何使用python虚拟环境，非常简单，文档最下面提供Windows系统和Linux系统创建和使用虚拟环境的方法


### Python安装 pytorch-cpu版本yolo8（Linux建议Python3.8，Windows建议Python3.10）
* pip install ultralytics==8.2.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.2 torchvision==0.16.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

### Python安装 pytorch-cuda版本yolo8（Linux建议Python3.8，Windows建议Python3.10）
* pip install ultralytics==8.2.86 -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
* 注意：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择


### 快速开始
~~~
//查看已安装的yolo8版本
yolo -V

//训练检测模型（gpu版本）
yolo detect train model=yolov8n.pt data=/datasets/data.yaml batch=64 epochs=5 imgsz=640 device=cuda
（注意：关于/datasets/data.yaml文件，是指向训练数据集的配置文件，如果不清楚可以先下载一份数据集参考下）

//训练检测模型（cpu版本）
yolo detect train model=yolov8n.pt data=/datasets/data.yaml batch=64 epochs=5 imgsz=640 device=cpu

//测试模型
yolo detect predict model=runs/train/best.pt source=test.jpg

//将pt模型转换为onnx格式模型
//依赖库：pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=best.pt format=onnx

//将pt模型转换为openvino格式模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=best.pt format=openvino

~~~


### mo命令将pt模型转换为openvino模型（方式二）
* mo命令是openvino官方提供的模型转换工具
* mo参考文档 https://blog.csdn.net/qq_44632658/article/details/131270531
~~~
//安装mo命令行，将onnx转换为openvino模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
yolo export model=yolov8n.pt format=onnx
mo --input_model yolov8n.onnx  --output_dir yolov8n_openvino_model
~~~

### trtexec命令将pt模型转换为tensorrt模型
~~~
//将pt模型转换为onnx格式模型
yolo export model=yolov8n.pt format=onnx

//将onnx模型转换为tensorrt格式模型(在ovtrt版本的视频行为分析系统的xcms_core文件夹下面有trtexec工具)
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n.fp16.engine --fp16
~~~

### yolo8训练相关文档
* yolo8官方训练参数：https://docs.ultralytics.com/zh/modes/train/#train-settings
* yolo8训练参数参考: https://blog.csdn.net/weixin_51692073/article/details/133875143
* yolo8开源地址：https://github.com/ultralytics/ultralytics
* yolo8官方模型-夸克网盘下载地址：https://pan.quark.cn/s/59a17310b205

### 训练数据集（免费下载）
* 训练数据集-夸克网盘下载地址：https://pan.quark.cn/s/5dcc2f724bcc
* 检测抽烟数据集20241012
* 检测打架数据集20241012
* 检测反光衣数据集20241013
* 检测粉尘数据集20241013
* 检测火焰烟雾数据集20241012
* 检测人体5动作-站着-摔倒-坐-深蹲-跑数据集20241012
* 检测人头和安全帽数据集20241013
* 检测睡岗数据集20241210
* 检测学生3状态数据集20241022
### Windows系统安装Python虚拟环境
~~~

//创建虚拟环境
python -m venv venv

//切换到虚拟环境
venv\Scripts\activate

//更新虚拟环境的pip版本（可以不更新）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~

### Linux系统安装Python虚拟环境

~~~
//创建虚拟环境
python -m venv venv

//激活虚拟环境
source venv/bin/activate

//更新虚拟环境的pip版本（可以不更新）
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~



