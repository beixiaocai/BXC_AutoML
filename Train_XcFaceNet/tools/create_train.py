"""
训练数据集生成train.txt文件的脚本

"""
import os

if __name__ == "__main__":
    train_path  = "../datasets/train"  # 训练数据集
    train_desc_path = "../datasets/train.txt" # 训练数据集描述文件

    if os.path.exists(train_desc_path):
        os.remove(train_desc_path)

    types_name  = os.listdir(train_path)
    types_name  = sorted(types_name)
    train_desc_f = open(train_desc_path, 'w')

    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(train_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)
        for photo_name in photos_name:
            train_desc_f.write(str(cls_id) + ";" + '%s'%(os.path.join(os.path.abspath(train_path), type_name, photo_name)))
            train_desc_f.write('\n')
    train_desc_f.close()
