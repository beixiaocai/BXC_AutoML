import os
import shutil
from datetime import datetime

"""
批量将指定文件夹的所有.jpg结尾的文件夹拷贝到目标文件夹
"""


def handle_0InnerFolder(src_dir, dst_dir,dst_filename_prefix=""):
    print("handle_0InnerFolder() start:", datetime.now())

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    filenames = os.listdir(src_dir)
    count = 0
    for filename in filenames:
        if filename.endswith(".jpg"):
            count += 1

            print("------------[count=%d] filename=%s------------" % (count, filename))
            src_filepath = os.path.join(src_dir, filename)
            dst_filename = "%s%d.jpg" % (dst_filename_prefix, count)
            dst_filepath = os.path.join(dst_dir, dst_filename)

            if os.path.exists(dst_filepath):
                print("目标文件名已经存在:%s" % dst_filepath)
                os.remove(dst_filepath)
            shutil.copy(src_filepath, dst_filepath)
        else:
            print("不合法的文件名：%s" % filename)

    print("handle_0InnerFolder() end:", datetime.now())

def handle_2InnerFolder(src_dir, dst_dir,dst_filename_prefix="",inner_count=1):
    print("handle_2InnerFolder() start:", datetime.now())

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    count = 0
    folders = os.listdir(src_dir)
    for folder in folders:
        folder_path = os.path.join(src_dir, folder)
        folder_folders = os.listdir(folder_path)
        if len(folder_folders) == 1:
            folder_folder_path = os.path.join(folder_path, folder_folders[0])
            filenames = os.listdir(folder_folder_path)
            for filename in filenames[0:inner_count]:
                if filename.endswith(".jpg"):
                    src_filepath = os.path.join(folder_folder_path, filename)
                    dst_filename = "%s_%s_%d.jpg" % (dst_filename_prefix, folder,count)
                    dst_filepath = os.path.join(dst_dir, dst_filename)
                    if os.path.exists(dst_filepath):
                        print("目标文件名已经存在:%s" % dst_filepath)
                        os.remove(dst_filepath)
                    shutil.copy(src_filepath, dst_filepath)
                    count += 1

    print("handle_2InnerFolder() end:", datetime.now())




if __name__ == '__main__':
    # handle_0InnerFolder(
    #     src_dir="F:\\ai\\data\\buy\\Z8300YOLO\\val\\images",
    #     dst_dir="F:\\ai\\data\\buy\\Z8300_val_seg",
    #     dst_filename_prefix="seg20250730"
    # )
    handle_2InnerFolder(
        src_dir="F:\\ai\\data\\20250712factory\\素材_0809_标注3分类箱子分割样本-待标注\\data0809",
        dst_dir="F:\\ai\\data\\20250712factory\\素材_0809_标注3分类箱子分割样本-待标注\\snapshot",
        dst_filename_prefix="ss0809",inner_count=1
    )

    handle_2InnerFolder(
        src_dir="F:\\ai\\data\\20250712factory\\素材_0809_标注3分类箱子分割样本-待标注\\data0811",
        dst_dir="F:\\ai\\data\\20250712factory\\素材_0809_标注3分类箱子分割样本-待标注\\snapshot",
        dst_filename_prefix="ss0811",inner_count=3
    )