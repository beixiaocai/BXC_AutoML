import shutil
import os

"""
将负样本的图片文件夹转换为yolo数据集
"""

def handle(negative_image_dir, dst_detect_dir):

    dst_detect_images_dir = os.path.join(dst_detect_dir, "images")
    dst_detect_labels_dir = os.path.join(dst_detect_dir, "labels")

    if not os.path.exists(dst_detect_images_dir):
        os.makedirs(dst_detect_images_dir)
    if not os.path.exists(dst_detect_labels_dir):
        os.makedirs(dst_detect_labels_dir)

    i = 0
    filenames = os.listdir(negative_image_dir)
    print("negative_image_dir:",negative_image_dir,len(filenames),filenames)

    for filename in filenames:
        name = None
        names = filename.split(".")

        if len(names) == 2:
            name = names[0]
            print("parse1 success filename=%s,name=%s,len(names)=%d " % (filename, name, len(names)))
        else:
            if filename.endswith(".jpg"):
                name = filename[0:-4]
                print("parse2 success filename=%s,name=%s,len(names)=%d " % (filename, name, len(names)))

        if name:
            src_image_path = os.path.join(negative_image_dir, name+".jpg")
            dst_image_path = os.path.join(dst_detect_images_dir, name+".jpg")

            dst_label_path = os.path.join(dst_detect_labels_dir, name+".txt")
            try:
                shutil.copyfile(src_image_path, dst_image_path)
                print("--------%d---------" % i)
                print("src_image_path=", src_image_path)
                print("dst_image_path=", dst_image_path)

                dst_label_f = open(dst_label_path, "w")
                dst_label_f.close()


            except Exception as e:
                print("copy失败：",e,src_image_path)
                try:
                    os.remove(dst_image_path)
                except: pass
                try:
                    os.remove(dst_label_path)
                except: pass
        else:
            print("parse error filename=%s,len(names)=%d " % (filename, len(names)))

        i += 1

if __name__ == '__main__':
    handle(
        negative_image_dir="F:\\ai\\data\\20250712factory\\source0724b_cartons3\\alarm",
        dst_detect_dir="F:\\ai\\data\\20250712factory\\source0724b_cartons3\\alarm_detect\\train"
    )