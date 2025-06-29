### Train_AudioNet
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
//训练模型
python train.py --data_dir data --save_dir checkpoints --batch_size 32 --epochs 30 --lr 0.001
    
//测试模型（方式1）
python test.py --model_path  checkpoints/best_model.pt --audio_path data/val/non_snore/0_400.wav
//测试模型（方式2）
python test_ov_by_torchaudio.py --model_path  checkpoints/best_model_openvino_model/best_model.xml --audio_path data/val/non_snore/0_400.wav
//测试模型（方式3）
python test_ov_by_rosa.py --model_path  checkpoints/best_model_openvino_model/best_model.xml --audio_path data/val/non_snore/0_400.wav

//将pt模型转换为onnx格式的模型
//依赖库：pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
python export2onnx.py --pt checkpoints/best_model.pt --onnx checkpoints/best_model.onnx 


//将onnx模型转换为openvino格式的模型
//依赖库：pip install openvino==2024.3.0 openvino-dev==2024.3.0 onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
mo --input_model checkpoints/best_model.onnx  --output_dir checkpoints/best_model_openvino_model


~~~

### 训练数据集（免费下载）
* 打鼾+不打鼾语音识别2分类数据集-夸克网盘下载地址：https://pan.quark.cn/s/4d83dabff0a6



