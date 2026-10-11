# onnx2rknn
* 在x86或arm架构的处理器上，将onnx模型转换为rknn模型
* 定位说明：ResNet / XcFaceNet / DeepSort 等模型的 rknn 转换仍使用本模块；YOLO 检测模型推荐 Train_yolo26 原生导出 rknn（训练与转换同一框架，无需本模块），本模块的 YOLO 转换脚本仅服务于 old/ 目录的历史 rk 版模型

### 第一步（安装模型转换环境）
* 如下提供三种安装模型转换环境的方式，任选其一

#### 安装方式1：在x86安装docker版模型转换环境
~~~

//构建镜像
docker build -f x86-dockerfile -t x86-onnx2rknn:20250329 .

//镜像导出
docker save -o x86-onnx2rknn.20250329.tar  x86-onnx2rknn:20250329

//启动镜像
docker run -d -i -t -u root --privileged -e LANG=C.UTF-8 x86-onnx2rknn:20250329 /bin/bash

//用户你好，如果不想自行构建镜像，可以下载作者上传到网盘中的该镜像文件，下载镜像后，导入到docker，直接启动即可
//镜像下载链接：https://pan.quark.cn/s/e1c27069e57e
（请下载x86-onnx2rknn.20250329.zip，解压后获得x86-onnx2rknn.20250329.tar）


//镜像导入
docker load -i x86-onnx2rknn.20250329.tar

//启动镜像
docker run -d -i -t -u root --privileged -e LANG=C.UTF-8 x86-onnx2rknn:20250329 /bin/bash

~~~

#### 安装方式2：在x86（Linux）安装模型转换环境
~~~
//首先，对于新手而言，强烈建议使用Ubuntu20.04，并确保系统默认Python版本是Python3.8
并不是不支持其它系统或其他Python版本，主要是因为Python依赖库版本很容易不一致，不一致后虽然容易解决，但新手似乎不具备这个能力

//创建虚拟环境
python3 -m venv venv

//切换到虚拟环境
source venv/bin/activate

//在虚拟环境安装依赖库
pip install --retries 60  -r 3rdparty-x86/requirements_cp38-2.3.0.txt
pip install 3rdparty-x86/rknn_toolkit2-2.3.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

~~~

#### 安装方式3：在arm（Linux）安装模型转换环境

~~~
//首先，对于新手而言，强烈建议使用Ubuntu20.04，并确保系统默认Python版本是Python3.8
并不是不支持其它系统或其他Python版本，主要是因为Python依赖库版本很容易不一致，不一致后虽然容易解决，但新手似乎不具备这个能力

//创建虚拟环境
python3 -m venv venv

//切换到虚拟环境
source venv/bin/activate

//在虚拟环境安装依赖库
pip install --retries 60  -r 3rdparty-arm/arm64_requirements_cp38.txt
pip install 3rdparty-arm/rknn_toolkit2-2.3.0-cp38-cp38-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

~~~

### 第二步（模型转换：onnx模型->rknn模型）
* 首先，看到本文档的你，请务必清楚的知道：用于转换rknn的onnx，并不是通用的onnx。按模型类型分三种情况：
  - **YOLO 检测（新方案，推荐）**：用 Train_yolo26 训练，`yolo export format=rknn` 一条命令完成"适配ONNX导出 + rknn转换"，见 Train_yolo26/export_rknn.md，无需本模块
  - **YOLO 检测（历史方案）**：onnx 必须来自 old/ 目录的 rk 版训练框架（魔改源码版），再用下方 onnx2rknn_Yolo.py 转换，仅作历史保留
  - **其他模型（ResNet / XcFaceNet / DeepSort 等）**：训练框架导出 onnx 后，用本模块对应脚本转换
* 接下来，如果你已经训练并获得了可用于转换rknn的onnx，那就开始进行模型转换吧
~~~
//将old目录 rk 版 yolo5/yolo8/yolo11 的onnx转换为rknn（历史方案，onnx_filename 和 platform 在脚本内修改）
python3 onnx2rknn_Yolo.py 

//将其他模型的onnx转换为rknn（onnx文件名与目标平台在脚本内修改）
python3 onnx2rknn_ResNet.py
python3 onnx2rknn_XcFaceNet.py
python3 onnx2rknn_DeepSort.py

~~~
