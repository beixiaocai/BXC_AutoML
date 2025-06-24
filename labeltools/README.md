# labeltools
* 样本处理工具

### 环境依赖

| 程序         | 版本      |
| ---------- | ------- |
| python     | 3.8+    |
| 依赖库      | requirements.txt |

### 安装依赖库
    * pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


### 如何使用
~~~

//labelme格式的样本转resnet分类格式样本
python dataset_labelme_json2classify.py

//resnet分类格式样本按指定比例分割成训练集和测试集
python dataset_classify2classify.py

//labelme格式的样本转yolo检测格式样本
python dataset_labelme_json2detect.py

//yolo检测格式样本按指定比例分割成训练集和测试集
python dataset_detect2detect.py

//yolo检测格式样本转resnet分类格式样本
python dataset_detect2classify.py

//voc格式样本转yolo检测格式样本
python dataset_voc2detect.py

~~~