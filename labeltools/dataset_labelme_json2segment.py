import json
import os
import time
import shutil

g_label2index = {
    "water": 0
}


def handle(labelme_dir, detect_dir):
    filenames = os.listdir(labelme_dir)
    print("handle() labelme_dir=%s,len(filenames)=%d" % (labelme_dir, len(filenames)))

    detect_images_dir = os.path.join(detect_dir, "images")
    detect_labels_dir = os.path.join(detect_dir, "labels")
    if not os.path.exists(detect_images_dir):
        os.makedirs(detect_images_dir)
    if not os.path.exists(detect_labels_dir):
        os.makedirs(detect_labels_dir)

    flag = "random" + str(int(time.time()))
    index = 0
    for filename in filenames:
        if filename.endswith(".json"):
            names = filename.split(".")
            if len(names) == 2:
                name = names[0]
                print("开始处理第%d张图片%s" % (index, filename))
                # try:
                json_filepath = os.path.join(labelme_dir, filename)
                f = open(json_filepath, "r")
                content = f.read()
                f.close()

                json_data = json.loads(content)
                # version = json_data.get("version")
                shapes = json_data.get("shapes")
                imagePath = json_data.get("imagePath")
                imageWidth = json_data.get("imageWidth")
                imageHeight = json_data.get("imageHeight")

                imagePath_abs = os.path.join(labelme_dir, imagePath)
                if os.path.exists(imagePath_abs) and len(shapes) > 0:
                    j = 0
                    save_name = "%s-%d" % (flag, index)
                    save_image_filepath = os.path.join(detect_images_dir, save_name + ".jpg")
                    save_label_filepath = os.path.join(detect_labels_dir, save_name + ".txt")

                    success_count = 0
                    try:
                        save_label_f = open(save_label_filepath, "w")
                        for shape in shapes:
                            label = shape.get("label")
                            shape_type = shape.get("shape_type")
                            points = shape.get("points")
                            # print(label, shape_type, points)

                            points_len = len(points)
                            if points_len >= 2:
                                label_index = g_label2index.get(label, None)
                                if label_index is None:
                                    print("\t未定义的标签名:json_filepath=%s" % json_filepath)
                                else:
                                    save_label_f.write("%d" % label_index)
                                    success_count += 1
                                    for i in range(points_len):
                                        x = float(points[i][0])
                                        y = float(points[i][1])

                                        if 0 <= x <= imageWidth and 0 <= y <= imageHeight:
                                            x_ratio = x / float(imageWidth)
                                            y_ratio = x / float(imageHeight)

                                            line_content = " %.6f %.6f" % (x_ratio, y_ratio)
                                            save_label_f.write(line_content)
                                        else:
                                            print("\t目标框超过了背景范围")
                                    save_label_f.write("\n")

                            j += 1
                        save_label_f.close()


                    except Exception as e:
                        print("处理图片时发生错误：", e)

                    if success_count > 0:
                        shutil.copyfile(imagePath_abs, save_image_filepath)
                    else:
                        print("图片(%s)无目标框，删除label文件: " % imagePath_abs)
                        try:
                            os.remove(save_label_filepath)
                        except:
                            pass

                # except Exception as e:
                #     print("\t报错：第%d张图片%s" % (index, filename), e)

                index += 1


def handle_parent(labelme_parent_dir, detect_dir):
    print("handle_parent() start")

    dir_names = os.listdir(labelme_parent_dir)
    print("handle_parent() labelme_parent_dir=%s,len(dir_names)=%d" % (labelme_parent_dir, len(dir_names)))
    for dir_name in dir_names:
        labelme_dir = os.path.join(labelme_parent_dir, dir_name)
        if os.path.isdir(labelme_dir) and not dir_name.startswith("__"):
            handle(labelme_dir=labelme_dir, detect_dir=detect_dir)


if __name__ == '__main__':
    print("__main__")

    handle(
        labelme_dir="C:\\Users\\fufu\\Desktop\\action_video\\20240821waterlevel",
        detect_dir="D:\\datasets\\bxc_segment_sample\\20240821waterlevel\\train"
    )
