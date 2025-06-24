import os
import shutil
from datetime import datetime

"""
批量对文件夹内部图片进行重置序号
"""


def handle(src_dir, prefix=""):
    print("批处理重置文件序号开始:", datetime.now())

    filenames = os.listdir(src_dir)
    saveCount = 0
    for filename in filenames:
        if filename.endswith(".jpg"):
            saveCount += 1

            print("正在处理%s" % filename)
            src_filepath = os.path.join(src_dir, filename)
            new_filename = "%s%d.jpg" % (prefix, saveCount)
            new_filepath = os.path.join(src_dir, new_filename)
            if os.path.exists(new_filepath):
                print("被修改的文件名已经存在:%s，提前退出程序" % new_filepath)
                return
            shutil.copy(src_filepath, new_filepath)
            os.remove(src_filepath)


        else:
            print("不合法的文件名：%s" % filename)

    print("批处理重置文件序号结束:", datetime.now())


if __name__ == '__main__':
    handle(
        src_dir="D:\\datasets\\sample\\smoke_group\\smoke20240820001",
        prefix="xy"
    )
