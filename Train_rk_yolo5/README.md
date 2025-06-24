### Train_rk_yolo5
* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_AutoML
* github开源地址：https://github.com/beixiaocai/BXC_AutoML


### 第一步（安装模型训练环境）

~~~

//创建虚拟环境
python -m venv venv

//windows系统激活虚拟环境（建议python3.10，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）
venv\Scripts\activate

//linux系统激活虚拟环境（建议python3.8，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）
source venv/bin/activate

//安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

//（如果电脑没有英伟达显卡，可以安装cpu版pytorch2.1.0）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0

//（如果电脑有英伟达显卡，并且安装了显卡驱动，可以安装cuda版pytorch2.1.0加速）
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

~~~


### 第二步（训练模型）
~~~

//开启训练
python train.py  --weights yolov5n.pt --data /data/smoke/data.yaml --epochs 10  --batch-size 64 --device 0

请注意：/data/smoke/data.yaml就是对应样本数据

//测试模型示例1
python detect.py  --weights=yolov5n.pt --source=bus.jpg  --device=cpu

//测试模型示例2
python detect.py  --weights=/data/runs/train/exp/weights/best.pt --source=bus.jpg  --device=cpu

~~~

### 第三步（模型转换：pt模型->onnx模型）

~~~

//windows系统安装依赖库（建议python3.10，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）：
pip install onnxruntime==1.19.0 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple

//linux系统依赖库（建议python3.8，其他python的依赖库版本可能会有所不同，也可以根据报错自行修改版本）：
pip install onnxruntime==1.16.3 onnx==1.16.1  -i https://pypi.tuna.tsinghua.edu.cn/simple

//重要的一步，将rk版yolo5训练的pt模型转换为rk设备支持的onnx模型
python export.py --rknpu --weight yolov5n.pt

~~~


### 第四步（模型转换：onnx模型->rknn模型）

~~~
//将rk设备支持的onnx转换为rknn模型（参考文档）
https://gitee.com/Vanishi/BXC_AutoML/tree/master/onnx2rknn

~~~


### 其他
* rk版yolo5代码和yolo5官方代码训练的模型是不通用的
* 务必不要从yolo5的github官方仓库下载代码！！！本模块的代码来自于 https://github.com/airockchip/yolov5
* https://github.com/airockchip/yolov5 来自于 https://github.com/ultralytics/yolov5 

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