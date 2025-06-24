### Train_ResNet
* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_AutoML
* github开源地址：https://github.com/beixiaocai/BXC_AutoML

### 安装环境请注意
* 在使用python开发项目时，推荐使用python的虚拟环境。因为同一台电脑上很可能会安装多个python项目，而不同的python项目可能会使用不同的依赖库，为了避免依赖库不同而导致的冲突，可以使用python虚拟环境
* 关于如何使用python虚拟环境，其实非常简单，文档最下面提供Windows系统和Linux系统创建和使用虚拟环境的方法
* Windows建议使用Python3.10，Linux建议使用Python3.8
* [python官网下载地址](https://www.python.org/getit/)
* [python网盘下载地址](https://pan.quark.cn/s/72df133d1343)

#### Windows系统安装Python虚拟环境
~~~

//创建虚拟环境
python -m venv venv

//切换到虚拟环境
venv\Scripts\activate

//更新虚拟环境的pip版本
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

//在虚拟环境中安装依赖库
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~

#### Linux系统安装Python虚拟环境

~~~
//创建虚拟环境
python -m venv venv

//激活虚拟环境
source venv/bin/activate

//更新虚拟环境的pip版本
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

//在虚拟环境中安装依赖库
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

~~~

### 安装pytorch-cpu依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 安装pytorch-gpu依赖库
* pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
* 注意：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择


### 如何使用
~~~
//训练模型（gpu版本）
python train.py --epochs 90 --batch-size 64 --workers 0 --device cuda --checkpoint ./checkpoint --data-dir ./data

//基于已经训练的模型再次训练（gpu版本）
python train.py --epochs 90 --batch-size 64 --workers 0 --device cuda --checkpoint ./checkpoint --data-dir ./data --resume ./checkpoint/model.pth

//训练模型（cpu版本）
python train.py --epochs 90 --batch-size 64 --workers 0 --checkpoint ./checkpoint --data-dir ./data

//基于已经训练的模型再次训练（cpu版本）
python train.py --epochs 90 --batch-size 64 --workers 0  --checkpoint ./checkpoint --data-dir ./data --resume ./checkpoint/model.pth

//测试模型（gpu版本）
python test.py --test-dir ../data/test --checkpoint ./checkpoint/model.pth --device cuda

//测试模型（cpu版本）
python test.py --test-dir ./data/test --checkpoint ./checkpoint/model.pth

//将pt模型转换为onnx格式的模型（gpu版本）
//依赖库：pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
python export2onnx.py --pt ./checkpoint/model.pt --onnx ./checkpoint/model.onnx --device cuda

//将pt模型转换为onnx格式的模型（cpu版本）
//依赖库：pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
python export2onnx.py --pt ./checkpoint/model.pt --onnx ./checkpoint/model.onnx

//将onnx模型转换为openvino格式的模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
mo --input_model ./checkpoint/model.onnx  --output_dir ./checkpoint/model_openvino_model
mo --input_model ./checkpoint/model.onnx  --output_dir ./checkpoint/model_openvino_model --input_shape [1..64,3,224,224]

注意: mo命令是openvino官方提供的模型转换工具，mo使用可参考 https://blog.csdn.net/qq_44632658/article/details/131270531
显式禁用压缩: --compress_to_fp16=False


//将onnx模型转换为tensorrt格式的模型
trtexec --onnx=./checkpoint/model.onnx --saveEngine=best.fp16.engine --fp16 --minShapes=input:1x3x224x224   --optShapes=input:64x3x224x224   --maxShapes=input:64x3x224x224
注意: trtexec命令tensorrt官方提供的模型转换工具，trtexec使用可参考 https://blog.csdn.net/weicao1990/article/details/130827602

~~~

### 训练数据集（免费下载）
* 猫狗2分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/982dd16cb29d
* 车型9分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/f698d0e99a4b

