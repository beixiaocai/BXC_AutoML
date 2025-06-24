### Train_PlateNet
* 作者：北小菜 
* 官网：http://www.beixiaocai.com
* 邮箱：bilibili_bxc@126.com
* QQ：1402990689
* 微信：bilibili_bxc
* 哔哩哔哩主页：https://space.bilibili.com/487906612
* gitee开源地址：https://gitee.com/Vanishi/BXC_AutoML
* github开源地址：https://github.com/beixiaocai/BXC_AutoML

### 介绍
* 本项目是基于PaddleOCR框架，在PP-OCRv4的开源模型基础上，使用开源的车牌识别样本集CCPD2020迭代训练而来。本项目的目的是，通过编写详细的文档，尽可能让所有人都能学会训练车牌识别模型和调用车牌识别模型。降低行业门槛。

* 项目根目录下models文件夹里面内置了作者训练好的车牌检测和车牌识别模型，大家如果不想训练，可以直接使用，调用方法参考tests.py

* 在视频行为分析系统v4接入方案：https://gitee.com/Vanishi/algorithm_examples/blob/master/api_behaviour/api_behaviour_ocr_plate.py

### 安装训练环境
~~~
（1）首先确保您已经具备Python运行环境
目前支持 Python 3.8 至 Python 3.12（Windows系统推荐Python3.10，作者在Windows用的是3.10）

（2）安装paddlepaddle
//安装CPU版本
python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

//安装GPU版本（cu118），需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

//安装GPU版本（cu123），需显卡驱动程序版本 ≥545.23.06（Linux）或 ≥545.84（Windows）
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

（3）安装paddleocr
//注意：截止到目前2025/06/22，paddleocr已经推出3.0，相比于此前的2.0版本改变很大，
本项目是基于2.10.0和PP-OCRv4完成的，如果大家选择paddleocr3.0，则可能会面临很多不确定性，建议保持一致

pip install paddleocr==2.10.0


~~~

### 下载PP-OCRv4的开源预训练基础模型 + 训练样本集
~~~
（1）下载PP-OCRv4的开源预训练基础模型 + 下载作者基于CCPD2020转换后适合paddleocr直接训练的样本集
夸克网盘下载地址： https://pan.quark.cn/s/e0c5fefb8228
网盘文件介绍/
	├── pretrain_models.zip  # （需要下载）PP-OCRv4的开源预训练基础模型
		pretrain_models/
		├── ch_PP-OCRv4_det_cml_teacher_pretrained
		└── ch_PP-OCRv4_rec_train
    └── CCPD2020_PPOCR.zip   # （需要下载）CCPD2020转换后适合paddleocr直接训练的样本集
		CCPD2020_PPOCR/
		├── ccpd_green
		└── PPOCR 
		
tips: 网盘中上述文件除外，都不需要下载，是用来归档备份的。
	
注意：解压后的pretrain_models和CCPD2020_PPOCR请务必放置到项目根目录下

~~~

### 训练模型+验证模型+导出模型
~~~
（1）车牌识别主要分为两个过程，第一步是车牌检测过程（det），第二步是车牌识别过程（rec）
//在开始训练前，请前往configs文件夹，分别打开configs/plate_det.yml和configs/plate_rec.yml，里面涉及的数据集路径前缀改成与自己本机环境一致，注意路径不要包含中文


（2）车牌检测模型的训练+评估+导出
//（det）训练模型
python tools/train.py -c configs/plate_det.yml -o Global.pretrained_model=pretrain_models/ch_PP-OCRv4_det_cml_teacher_pretrained/teacher.pdparams


//（det）评估模型
python tools/eval.py -c configs/plate_det.yml -o Global.pretrained_model=output/plate_det/best_model/model.pdparams Eval.dataset.data_dir=CCPD2020_PPOCR/ccpd_green Eval.dataset.label_file_list=[CCPD2020_PPOCR/PPOCR/test/det.txt]


//（det）导出模型
python tools/export_model.py -c configs/plate_det.yml -o Global.pretrained_model=output/plate_det/best_model/model.pdparams Global.save_inference_dir=output/plate_det/inference_model

（3）车牌识别模型的训练+评估+导出
//（rec）训练模型
python tools/train.py -c configs/plate_rec.yml -o Global.pretrained_model=pretrain_models/ch_PP-OCRv4_rec_train/student.pdparams


//（rec）评估模型
python tools/eval.py -c configs/plate_rec.yml -o Global.pretrained_model=output/plate_rec/best_model/model.pdparams Eval.dataset.data_dir=CCPD2020_PPOCR/PPOCR Eval.dataset.label_file_list=[CCPD2020_PPOCR/PPOCR/test/rec.txt]


//（rec）导出模型
python tools/export_model.py -c configs/plate_rec.yml -o Global.pretrained_model=output/plate_rec/best_model/model.pdparams Global.save_inference_dir=output/plate_rec/inference_model


补充介绍：关于训练模型结构和导出模型结构的区别

//直接训练的模型结构
output/plate_det/best_model/
    ├── model.pdopt    # 优化器状态（用于恢复训练）
    └── model.pdparams # 模型参数（权重和偏置）

//训练的模型导出后的结构
output/plate_det/inference_model/
    ├── inference.pdmodel    # 模型结构
    ├── inference.pdiparams  # 模型参数
    └── inference.pdiparams.info  # 参数信息

~~~

### 测试图片
* 测试图片的模型是导出后的模型，直接训练的模型无法直接用来推理
~~~
//编辑 tests.py，修改模型和测试图片的相关地址，即可执行测试
python tests.py

~~~

### 关于如何部署或移植到项目中

* 大家可以参考上述步骤tests.py文件，将推理实现代码移植到自己的项目中使用。 如果是其他编程语言调用，建议使用python创建一个车牌识别的http server，让其他语言通过接口调用