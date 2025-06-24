import shutil
import os

def handle(src_detect_dir, dst_detect_dir, freq=5):
    src_detect_images_dir = os.path.join(src_detect_dir, "images")
    src_detect_labels_dir = os.path.join(src_detect_dir, "labels")

    dst_detect_images_dir = os.path.join(dst_detect_dir, "images")
    dst_detect_labels_dir = os.path.join(dst_detect_dir, "labels")

    if not os.path.exists(dst_detect_images_dir):
        os.makedirs(dst_detect_images_dir)
    if not os.path.exists(dst_detect_labels_dir):
        os.makedirs(dst_detect_labels_dir)

    i = 0
    filenames = os.listdir(src_detect_images_dir)
    print(len(filenames),filenames)

    for filename in filenames:
        if i % freq == 0:
            name = None
            names = filename.split(".")

            if len(names) == 2:
                name = names[0]
                print("parse1:", len(names), "name=", name)
            else:
                if filename.endswith(".jpg"):
                    name = filename[0:-4]
                print("parse2:", len(names), "name=", name)

            if name:
                src_image_path = os.path.join(src_detect_images_dir, name+".jpg")
                src_label_path = os.path.join(src_detect_labels_dir, name+".txt")

                dst_image_path = os.path.join(dst_detect_images_dir, name+".jpg")
                dst_label_path = os.path.join(dst_detect_labels_dir, name+".txt")

                try:
                    shutil.copyfile(src_image_path, dst_image_path)
                    shutil.copyfile(src_label_path, dst_label_path)

                    print("--------%d---------" % i)
                    print("src_image_path=", src_image_path)
                    print("src_label_path=", src_label_path)
                    os.remove(src_image_path)
                    os.remove(src_label_path)
                except Exception as e:
                    print("copy失败：",e,src_image_path)
                    try:
                        os.remove(src_image_path)
                    except: pass
                    try:
                        os.remove(src_label_path)
                    except: pass
                    try:
                        os.remove(dst_image_path)
                    except: pass
                    try:
                        os.remove(dst_label_path)
                    except: pass
            else:
                print("filename=%s format error" % str(filename))

        i += 1

if __name__ == '__main__':
    # 将训练样本按照指定频率拆分一部分到测试样本
    handle(
        src_detect_dir="D:\\datasets\\bxc_detect_sample\\knife_gun_group\\train",
        dst_detect_dir="D:\\datasets\\bxc_detect_sample\\knife_gun_group\\valid",
        freq=6
    )