# cnnlstm_server

### 环境依赖

| 程序         | 版本      |
| ---------- | ------- |
| python     | 3.10+    |
| 依赖库      | requirements.txt |

### 安装 pytorch-cpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

### 安装 pytorch-gpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
* 注意：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择

### 如何启动服务
~~~
//启动服务
python cnnlstm_server.py

//调用接口地址
http://127.0.0.1:9710/algorithm

~~~