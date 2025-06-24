import shutil
import os
import time

def handle(src_dir, dst_dir,flag = None):
    if not os.path.exists(src_dir):
        print("源检测样本文件夹不存在:%s"%src_dir)
        return
    # if os.path.exists(dst_dir):
    #     shutil.rmtree(dst_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    if flag is None:
        flag = "flag%d" % int(time.time())


    src_images_dir = os.path.join(src_dir, "images")
    src_labels_dir = os.path.join(src_dir, "labels")

    dst_images_dir = os.path.join(dst_dir, "images")
    dst_labels_dir = os.path.join(dst_dir, "labels")

    if not os.path.exists(dst_images_dir):
        os.makedirs(dst_images_dir)
    if not os.path.exists(dst_labels_dir):
        os.makedirs(dst_labels_dir)

    saveCount = 0
    filenames = os.listdir(src_images_dir)
    if len(filenames) > 0:
        for filename in filenames:
            name = None
            if filename.endswith(".jpg"):
                name = filename[:-4]
            if name:
                saveCount += 1

                src_image_path = os.path.join(src_images_dir,filename)
                src_label_path = os.path.join(src_labels_dir,"%s.txt"%name)
                if os.path.exists(src_image_path) and os.path.exists(src_label_path):


                    # save_name = "%s_%s_%d" % (flag, name, saveCount)
                    save_name = "%s_%d" % (flag, saveCount)
                    dst_image_path = os.path.join(dst_images_dir, save_name + ".jpg")
                    dst_label_path = os.path.join(dst_labels_dir, save_name + ".txt")

                    print("--------------------%d------------------"%saveCount)
                    print(src_label_path,"-->",dst_label_path)
                    print(src_image_path,"-->",dst_image_path)

                    shutil.copy(src_label_path, dst_label_path)
                    shutil.copy(src_image_path, dst_image_path)


def handle_parent(src_parent_dir, dst_dir):
    print("handle_parent() start")

    dir_names = os.listdir(src_parent_dir)
    print("handle_parent() src_parent_dir=%s,len(dir_names)=%d" % (src_parent_dir, len(dir_names)))

    for dir_name in dir_names:
        src_dir = os.path.join(src_parent_dir, dir_name)
        if os.path.isdir(src_dir) and not dir_name.startswith("__"):
            #handle(src_dir=src_dir, dst_dir=dst_dir, flag=dir_name)
            handle(src_dir=src_dir, dst_dir=dst_dir, flag=None)



if __name__ == '__main__':


    handle(
        src_dir="E:\\datasets\\bxc_detect_sample_stand_fall_sit_squat_run\\train",
        dst_dir="E:\\datasets\\bxc_detect_sample_stand_fall_sit_squat_run2\\train"
    )
