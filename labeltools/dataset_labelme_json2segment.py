import os
import json
import shutil
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2


class LabelmeToYOLOv11Converter:
    def __init__(self, labelme_dir, output_dir, label_map, split_ratios=(0.8, 0.1, 0.1)):
        """
        :param labelme_dir: LabelMe标注文件所在目录（包含.json和对应图像）
        :param output_dir: YOLOv11格式数据集输出根目录
        :param label_map: 类别名称到ID的映射字典（如 {"cat": 0, "dog": 1}）
        :param split_ratios: 训练集/验证集/测试集划分比例（默认0.8:0.1:0.1）
        """
        self.labelme_dir = Path(labelme_dir)
        self.output_dir = Path(output_dir)
        self.label_map = label_map
        self.split_ratios = split_ratios

    def _create_dirs(self):
        """创建YOLOv11标准目录结构"""
        dirs = [
            "images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"
        ]
        for d in dirs:
            (self.output_dir / d).mkdir(parents=True, exist_ok=True)

    def _convert_single_annotation(self, json_path):
        """转换单个LabelMe标注文件为YOLOv11分割格式"""
        with open(json_path, 'r') as f:
            data = json.load(f)

        imagePath = data['imagePath']
        if imagePath.endswith('.jpg'):
            json_name = os.path.basename(json_path)
            imagePath = json_name.replace('.json', '.jpg')

        img_path = self.labelme_dir / imagePath
        if not img_path.exists():
            return None, "Image not found"

        img = cv2.imread(str(img_path))
        if img is None:
            return None, "Invalid image"
        h, w = img.shape[:2]

        yolo_lines = []
        for shape in data['shapes']:
            if shape['shape_type'] != 'polygon':
                print("不支持的形状类型：",shape['shape_type'])
                continue

            label = shape['label']

            print("label:",label)
            if label not in self.label_map:
                continue
            class_id = self.label_map[label]
            print("label:", label,"class_id:",class_id)
            # 计算边界框
            points = np.array(shape['points'])
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            width = (x_max - x_min) / w
            height = (y_max - y_min) / h

            # 归一化多边形点
            normalized_points = []
            for point in points:
                norm_x = point[0] / w
                norm_y = point[1] / h
                normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

            # 构建YOLO行（包含边界框和多边形）
            yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} " + " ".join(
                normalized_points)
            yolo_lines.append(yolo_line)

        return yolo_lines, "Success"

    def _generate_data_yaml(self):
        """生成YOLOv11训练必需的data.yaml配置文件"""
        content = f"path: {self.output_dir}\n"
        content += "train: images/train\n"
        content += "val: images/val\n"
        content += "test: images/test\n\n"
        content += "names:\n"
        for name, id in self.label_map.items():
            content += f"  {id}: {name}\n"

        with open(self.output_dir / 'data.yaml', 'w') as f:
            f.write(content)

    def convert_and_split(self):
        """执行完整转换流程"""
        self._create_dirs()
        json_files = [f for f in self.labelme_dir.glob('*.json')]
        random.shuffle(json_files)

        # 数据集划分
        n_total = len(json_files)
        n_train = int(n_total * self.split_ratios[0])
        n_val = int(n_total * self.split_ratios[1])
        train_files = json_files[:n_train]
        val_files = json_files[n_train:n_train + n_val]
        test_files = json_files[n_train + n_val:]

        # 转换函数
        def process_files(files, split_type):
            success_count = 0
            for json_path in tqdm(files, desc=f"Processing {split_type}"):
                base_name = json_path.stem
                img_path = json_path.with_suffix('.jpg')  # 支持.jpg/.png自动检测
                if not img_path.exists():
                    img_path = json_path.with_suffix('.png')

                # 复制图像
                if img_path.exists():
                    shutil.copy(
                        img_path,
                        self.output_dir / "images" / split_type / f"{base_name}{img_path.suffix}"
                    )
                else:
                    continue

                # 转换标签
                labels, status = self._convert_single_annotation(json_path)
                if labels:
                    with open(self.output_dir / "labels" / split_type / f"{base_name}.txt", 'w') as f:
                        f.write("\n".join(labels))
                    success_count += 1
            return success_count

        # 处理各数据集
        train_success = process_files(train_files, "train")
        val_success = process_files(val_files, "val")
        test_success = process_files(test_files, "test")

        # 生成配置文件
        self._generate_data_yaml()

        return {
            "total_files": n_total,
            "train": (len(train_files), train_success),
            "val": (len(val_files), val_success),
            "test": (len(test_files), test_success)
        }


# ===================== 使用示例 =====================
if __name__ == "__main__":
    # 配置参数
    converter = LabelmeToYOLOv11Converter(
        labelme_dir="F:\\ai\\data\\20250712factory\\label_0817_seg_merge",  # LabelMe原始数据目录
        output_dir="F:\\ai\\data\\20250712factory\\label_0817_seg_merge_yolo_seg",  # 输出目录（自动创建）
        label_map={
            "cartons": 0
        },  # 类别映射
        split_ratios=(0.6, 0.3, 0.1)  # 训练/验证/测试比例
    )

    # 执行转换
    results = converter.convert_and_split()

    # 打印结果
    print(f"\n{'=' * 50}\n转换完成！数据集结构已生成至: {converter.output_dir}")
    print(f"样本统计:")
    print(f"  - 总文件数: {results['total_files']}")
    print(f"  - 训练集: {results['train'][1]}/{results['train'][0]} (成功/总数)")
    print(f"  - 验证集: {results['val'][1]}/{results['val'][0]}")
    print(f"  - 测试集: {results['test'][1]}/{results['test'][0]}")
    print(f"配置文件: {converter.output_dir}/data.yaml")
    print(f"\n下一步: 直接使用以下命令训练YOLOv11模型:")
    print(f"yolo train-seg data={converter.output_dir}/data.yaml model=yolov11s-seg.yaml epochs=100 imgsz=640")