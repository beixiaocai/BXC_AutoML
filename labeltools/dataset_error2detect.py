import os


def copyFile(srcFilePath, dstFilePath):
    try:
        fr = open(file=srcFilePath, mode="rb")
        fw = open(file=dstFilePath, mode="wb")
        fw.write(fr.read())
        fw.close()
        fr.close()
    except Exception as e:
        print("copyFile error:%s" % str(e), srcFilePath, dstFilePath)


def handle():

    src_images_dir = os.path.join(src_dir, "images")
    src_labels_dir = os.path.join(src_dir, "txt-labels")

    dst_images_dir = os.path.join(dst_dir, "images")
    dst_labels_dir = os.path.join(dst_dir, "labels")
    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    if not os.path.exists(dst_labels_dir):
        os.makedirs(dst_labels_dir)

    __files = os.listdir(src_images_dir)
    i = 10000
    for __file in __files:
        i += 1
        __file_v = __file.split(".")
        if len(__file_v) == 2:
            name = __file_v[0]
            print("i=%d,图片=%s匹配到文本文件" % (i, "%s.jpg" % name))

            copyFile(
                srcFilePath=os.path.join(src_images_dir, "%s.jpg" % name),
                dstFilePath=os.path.join(dst_images_dir, "%d.jpg" % i)
            )

            copyFile(
                srcFilePath=os.path.join(src_labels_dir, "%s.txt" % name),
                dstFilePath=os.path.join(dst_labels_dir, "%d.txt" % i)
            )


        else:
            print("i=%d,图片=%s未匹配到文本文件" % (i, __file))


if __name__ == '__main__':
    src_dir = "E:\\project\\ai\\datasets\\20240525-buy-smoke\\783_smoke"
    dst_dir = "E:\\project\\ai\\datasets\\20240525-buy-smoke\\783_smoke_clean"

    handle()
