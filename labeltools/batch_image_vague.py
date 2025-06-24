import os
import cv2
import shutil
from datetime import datetime

"""
批量对文件夹内部图片进行模糊化处理
"""


def handle(src_dir, dst_dir, level=3):
    print("批处理图片模糊化开始:", datetime.now())
    if not os.path.exists(src_dir):
        print("批处理图片源文件夹不存在!")
        return

    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    filenames = os.listdir(src_dir)

    for filename in filenames:
        if filename.endswith(".jpg"):
            src_filepath = os.path.join(src_dir, filename)
            dst_filepath = os.path.join(dst_dir, filename)
            # 原图
            image = cv2.imread(src_filepath)

            for _ in range(level):
                image = cv2.blur(image, (5, 5))

            # sigmaX和sigmaY是高斯核函数的X和Y方向的标准差
            # sigmaX = 0
            # sigmaY = 0
            # # 高斯模糊
            # src_image_vague = cv2.GaussianBlur(image, (5, 5), sigmaX, sigmaY)

            cv2.imshow("image", image)
            cv2.imwrite(dst_filepath, image)

            cv2.waitKey(40)  # 1s内每帧延迟时间（毫秒）
        elif filename.endswith(".json"):
            src_filepath = os.path.join(src_dir, filename)
            dst_filepath = os.path.join(dst_dir, filename)
            shutil.copy(src_filepath, dst_filepath)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("批处理图片模糊化结束:", datetime.now())


if __name__ == '__main__':
    handle(
        src_dir="D:\\file\\images\\baideng",
        dst_dir="D:\\file\\images\\baideng_vague",
        level=3  # 模糊级别 0-100（数值越大，越模糊，0表示原图复制，1-100表示逐渐模糊）
    )
