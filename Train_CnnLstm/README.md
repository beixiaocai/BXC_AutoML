# Train_CnnLstm

### 环境依赖

| 程序         | 版本      |
| ---------- | ------- |
| python     | 3.10+    |
| 依赖库      | requirements.txt |

### 安装 pytorch-cpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple

### 安装 pytorch-gpu版本依赖库
* pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
* pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
* 注意：安装pytorch-gpu训练环境，请根据自己的电脑硬件选择cuda版本，比如我上面选择的https://download.pytorch.org/whl/cu121，并非适用所有电脑设备，请根据自己的设备选择


### 如何获取样本
~~~

//第一步，下载样本视频
下载地址：https://www.crcv.ucf.edu/data/UCF101.php

//第二步，下载样本视频后，强烈建议使用ucf101_tools文件夹里面的3个python脚本，依次执行python脚本，将样本视频转换成训练图片，转换完成后即可训练

~~~

### 如何训练模型
~~~

python train.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 1 --annotation_path E:\\ai\\datasets\\UCF-101_cnn_lstm\\annotation\\ucf101_01.json --video_path E:\\ai\\datasets\\UCF-101_cnn_lstm\\image_data  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 10



~~~

### 如何测试模型

~~~

python test.py  --annotation_path E:\\ai\\datasets\\UCF-101_cnn_lstm\\annotation\\ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes 10 --resume_path models/cnnlstm-epoch-100-acc-0.8533333333333334-loss-0.5009358089620417.pth

~~~
