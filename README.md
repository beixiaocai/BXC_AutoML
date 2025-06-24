### BXC_AutoML
* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_AutoML
* github开源地址：https://github.com/beixiaocai/BXC_AutoML

### 介绍
* Train_yolo8: yolo8官方训练框架
* Train_yolo11: yolo11官方训练框架
* Train_rk_yolo5: 适用于瑞芯微设备的yolo5模型训练框架
* Train_rk_yolo8: 适用于瑞芯微设备的yolo8模型训练框架
* Train_rk_yolo11: 适用于瑞芯微设备的yolo11模型训练框架
* onnx2rknn: 适用于瑞芯微设备的onnx模型转换为rknn模型工具
* Train_ResNet: 基于ResNet的图片分类算法训练框架
* Train_XcFaceNet: 基于MobileNet的人脸特征提取算法训练框架
* Train_PlateNet: 基于PaddleOCR框架和PP-OCRv4开源模型训练商用级高质量车牌识别模型
* Train_CnnLstm: 基于Cnn+Lstm的视频分类算法训练框架
* labeltools: 样本转换脚本

### 相关视频教程
* [训练第1讲，介绍算法训练工具](https://www.bilibili.com/video/BV1Em421s7ep)
* [训练第2讲，介绍收集样本/labelme标注样本/自动将标注的样本转换为yolo格式的训练数据集](https://www.bilibili.com/video/BV1cBHYekESt)
* [训练第3讲，介绍训练模型](https://www.bilibili.com/video/BV17WtjezEoQ)

### 相关工具
* [视频分割图片工具 hs](https://gitee.com/Vanishi/BXC_hs)
* [互联网图片样本下载工具 DownloadImage](https://gitee.com/Vanishi/BXC_DownloadImage)
* [图片标注样本工具 labelme](https://pan.quark.cn/s/7e6accce2a3e)
* [视频行为分析系统v4 xcms](https://gitee.com/Vanishi/xcms)

### 更新记录
#### 2025/06/24
* 新增Train_PlateNet
* 优化labeltools/dataset_detect_reset_detect.py
#### 2025/06/03
* 新增免费下载的训练数据集
#### 2025/04/30
* 新增Train_XcFaceNet
* Train_XcFaceNet是人脸特征提取算法的训练框架
* Train_XcFaceNet训练的人脸特征提取模型，可以非常简单的添加到视频行为分析系统v4进行使用
* onnx2rknn新增onnx2rknn_ResNet.py
#### 2025/04/29
* 更新Train_ResNet，优化纯cpu环境下训练模型的bug
#### 2025/04/05
* 更新Train_rk_yolo11/Train_rk_yolo8/Train_rk_yolo5安装文档，降低安装训练环境的难度，降低安装出错的可能性
#### 2025/03/28
* 更新onnx2rknn，新增支持docker版模型转换方式
* 优化Train_yolo8/README.md，Train_yolo11/README.md
#### 2025/01/06
* 新增Train_rk_yolo11
* onnx2rknn更新至rk2.3.0，并支持arm和x86两种架构
#### 2024/12/16
* 新增Train_yolo11
* 更新Train_yolo8/Train_yolo11/Train_ResNet推荐数据集链接，增加了更为优质的训练数据集链接
#### 2024/11/24 
* 新增适用于瑞芯微设备的yolo8模型训练框架Train_rk_yolo8
* 优化Train_rk_yolo5的使用说明
* 优化onnx2rknn的使用说明
#### 2024/9/11 
* 解决Train_yolo8将pt转onnx时错误问题，原因是onnx库的版本问题，在Train_yolo8/README.md文档已经设置了具体的版本
#### 2024/8/31 
* 解决Train_yolo8使用pip install ultralytics时错误问题，原因是python版本问题，在Train_yolo8/README.md文档已经设置了具体的版本
#### 2024/8/6
* （1）新增样本标注工具labelme的辅助脚本工具，可以将label标注的样本转化为yolo检测格式样本，或转化为resnet分类格式样本
* （2）新增检测格式样本和分类格式样本自动按照比例分割成训练集和测试集的脚本工具
#### 2024/7/30
* 新增基于cnn+lstm网络结构的视频分类算法训练框架，Train_CnnLstm
#### 2024/4/27 
* 首次上传

### 训练数据集（免费下载）
* 训练数据集-夸克网盘下载地址：https://pan.quark.cn/s/5dcc2f724bcc
* 检测攀爬数据集20250624
* 检测抽烟数据集20241012
* 检测打架数据集20241012
* 检测反光衣数据集20241013
* 检测粉尘数据集20241013
* 检测火焰烟雾数据集20241012
* 检测人体5动作-站着-摔倒-坐-深蹲-跑数据集20241012
* 检测人头和安全帽数据集20241013
* 检测睡岗数据集20241210
* 检测学生3状态数据集20241022
* 猫狗2分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/982dd16cb29d
* 车型9分类数据集-夸克网盘下载地址 https://pan.quark.cn/s/f698d0e99a4b