import os
import shutil
import argparse


def merge_labelme_datasets(source_root, target_dir):
    """
    合并labelme标注的分散数据集到单个文件夹
    新文件名格式：原文件夹名_原文件名

    参数:
        source_root (str): 包含多个样本集文件夹的根目录
        target_dir (str): 合并后的目标文件夹路径
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_dir, exist_ok=True)
    print(f"目标文件夹已创建: {target_dir}")

    # 用于跟踪文件名冲突的计数器
    file_counter = 1

    # 遍历源目录下的所有子文件夹
    for root, dirs, files in os.walk(source_root):
        # 跳过目标目录本身（防止自循环）
        if os.path.abspath(root) == os.path.abspath(target_dir):
            continue

        # 获取当前文件夹名称（将作为新文件名前缀）
        folder_name = os.path.basename(root)

        # 处理当前文件夹中的所有文件
        for filename in files:
            # 构建原始文件的完整路径
            src_path = os.path.join(root, filename)

            # 创建新文件名：文件夹名_原文件名
            new_filename = f"{folder_name}_{filename}"
            dest_path = os.path.join(target_dir, new_filename)

            # 处理文件名冲突（添加序号）
            conflict_count = 1
            while os.path.exists(dest_path):
                base, ext = os.path.splitext(new_filename)
                dest_path = os.path.join(target_dir, f"{base}({conflict_count}){ext}")
                conflict_count += 1

            # 移动并重命名文件
            shutil.copy2(src_path, dest_path)  # 使用copy2保留元数据
            print(f"已复制: {src_path} → {dest_path}")
            file_counter += 1

    print(f"\n合并完成! 共处理 {file_counter} 个文件")
    print(f"合并后目录: {target_dir}")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='合并labelme标注数据集')
    parser.add_argument('--source',  help='包含多个样本集的根目录路径')
    parser.add_argument('--target',  help='合并后的目标文件夹路径')

    args = parser.parse_args()

    args.source = "F:\\ai\\data\\20250712factory\\label_0817_seg"
    args.target = "F:\\ai\\data\\20250712factory\\label_0817_seg_merge"

    # 执行合并操作
    merge_labelme_datasets(
        source_root=args.source,
        target_dir=args.target
    )

