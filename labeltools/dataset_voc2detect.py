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
def writeFileContent(filepath,content):
    f = open(filepath, 'w', encoding="gbk")
    f.write(content)
    f.close()

def readFileContent(filepath):
    content = None
    for encoding in ["gbk", "utf-8"]:
        try:
            f = open(filepath, 'r', encoding=encoding)
            lines = f.readlines()
            f.close()
            if len(lines) == 1:
                top_line = lines[0]
                top_line = str(top_line).strip()
                line_v = top_line.split(" ")
                if len(line_v) == 5:
                    content = top_line
        except Exception as e:
            print("readFile %s error: encoding=%s,%s" % (str(filepath), encoding, str(e)))


    return content

def handle():
    src_images_dir = os.path.join(src_dir, "images")
    src_labels_dir = os.path.join(src_dir, "labels")

    dst_images_dir = os.path.join(dst_dir, "images")
    dst_labels_dir = os.path.join(dst_dir, "labels")

    inner_dirs = ["train","test","val"]
    for inner_dir in inner_dirs:

        __src_label_dir = os.path.join(src_labels_dir,inner_dir)
        __src_image_dir = os.path.join(src_images_dir,inner_dir)

        __dst_label_dir = os.path.join(dst_labels_dir,inner_dir)
        __dst_image_dir = os.path.join(dst_images_dir,inner_dir)


        if not os.path.exists(__dst_label_dir):
            os.makedirs(__dst_label_dir)
        if not os.path.exists(__dst_image_dir):
            os.makedirs(__dst_image_dir)

        __index = 1  # 清洗后的数据，序号从1开始
        __files = os.listdir(__src_label_dir)
        for __file in __files:
            __file_v = __file.split(".")
            if len(__file_v) == 2:
                __file = __file_v[0]
                __src_label_filepath = os.path.join(__src_label_dir,__file+".txt")
                __src_image_filepath = os.path.join(__src_image_dir,__file+".jpg")

                if os.path.exists(__src_label_filepath) and os.path.exists(__src_image_filepath):
                    # 满足label和image都存在
                    content = readFileContent(__src_label_filepath)
                    if content:
                        __dst_label_filepath = os.path.join(__dst_label_dir, str(__index) + ".txt")
                        __dst_image_filepath = os.path.join(__dst_image_dir, str(__index) + ".jpg")
                        copyFile(__src_label_filepath,__dst_label_filepath)
                        copyFile(__src_image_filepath,__dst_image_filepath)
                        __index += 1
                        print(inner_dir,__index)


if __name__ == '__main__':
    src_dir = "E:\\project\\ai\\datasets\\20240525-buy-smoke\\smoke_voc"
    dst_dir = "E:\\project\\ai\\datasets\\20240525-buy-smoke\\smoke_voc_clean"

    handle()
