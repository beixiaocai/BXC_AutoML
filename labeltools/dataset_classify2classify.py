import cv2
import time
import os


def handle(src_classify_dir, dst_classify_dir,freq=5):
    i = 0
    _dirs = os.listdir(src_classify_dir)
    for _dir in _dirs:
        src_dir_path = os.path.join(src_classify_dir, _dir)
        dst_dir_path = os.path.join(dst_classify_dir, _dir)
        if not os.path.exists(dst_dir_path):
            os.makedirs(dst_dir_path)

        filenames = os.listdir(src_dir_path)
        j = 0
        for filename in filenames:

            if j % freq == 0:
                try:
                    src_filepath = os.path.join(src_dir_path, filename)
                    dst_filepath = os.path.join(dst_dir_path, filename)
                    print("--------%d-%d---------" % (i, j))
                    print("src_filepath=", src_filepath)
                    print("dst_filepath=", dst_filepath)

                    rf = open(src_filepath,"rb")
                    content = rf.read()
                    rf.close()

                    wf = open(dst_filepath,"wb")
                    wf.write(content)
                    wf.close()

                    os.remove(src_filepath)

                    i += 1
                except Exception as e:
                    print(e)

            j += 1



if __name__ == '__main__':
    # 将训练样本按照指定频率拆分一部分到测试样本
    # handle(
    #     src_classify_dir="E:\\ai\\datasets\\bxc_classify_sample_person0805\\train",
    #     dst_classify_dir="E:\\ai\\datasets\\bxc_classify_sample_person0805\\val",
    #     freq=5
    # )
    handle(
        src_classify_dir="E:\\project\\custom\\20240930w_xo-700\\0923\\num\\train",
        dst_classify_dir="E:\\project\\custom\\20240930w_xo-700\\0923\\num\\val",
        freq=5
    )