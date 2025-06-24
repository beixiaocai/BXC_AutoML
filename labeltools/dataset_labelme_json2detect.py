import json
import os
import time
import shutil

g_label2index = {
    "person": 0,
    "bicycle": 1,
    # "gun": 1,
    # "dao": 2,
    # "baipai": 3
}

print("g_label2index:", len(g_label2index), g_label2index)
"""
g_index2labels = []
index = 0
while len(g_index2labels) < len(g_label2index)-1:
    for k, v in g_label2index.items():
        if v == index:
            g_index2labels.append(k)
            break
    index += 1

print("g_index2labels:",len(g_index2labels),g_index2labels)
"""


def handle(labelme_dir, detect_dir, flag=None):
    if flag is None:
        flag = "flag%d" % int(time.time())

    filenames = os.listdir(labelme_dir)
    print("handle() labelme_dir=%s,len(filenames)=%d" % (labelme_dir, len(filenames)))

    detect_images_dir = os.path.join(detect_dir, "images")
    detect_labels_dir = os.path.join(detect_dir, "labels")
    if not os.path.exists(detect_images_dir):
        os.makedirs(detect_images_dir)
    if not os.path.exists(detect_labels_dir):
        os.makedirs(detect_labels_dir)

    index = 0
    for filename in filenames:
        if filename.endswith(".json"):
            name = None
            names = filename.split(".")

            if len(names) == 2:
                name = names[0]
                print("parse1:", len(names), "name=", name)
            else:
                if filename.endswith(".json"):
                    name = filename[0:-5]
                print("parse2:", len(names), "name=", name)

            if name:
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
                    save_name = "%s_%s_%d" % (flag, name, index)
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

                            x1 = float(points[0][0])
                            y1 = float(points[0][1])
                            x2 = float(points[1][0])
                            y2 = float(points[1][1])

                            if x1 > 0 and y1 > 0 and x2 < imageWidth and y2 < imageHeight:

                                x_center = (x1 + x2) / 2 / float(imageWidth)
                                y_center = (y1 + y2) / 2 / float(imageHeight)
                                w = (x2 - x1) / float(imageWidth)
                                h = (y2 - y1) / float(imageHeight)

                                label_index = g_label2index.get(label, None)
                                if label_index is None:
                                    print("\t未定义的标签名:json_filepath=%s" % json_filepath)
                                else:
                                    line_content = "%d %.6f %.6f %.6f %.6f\n" % (label_index, x_center, y_center, w, h)
                                    save_label_f.write(line_content)
                                    success_count += 1
                            else:
                                print("\t目标框超过了背景范围")

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
            else:
                print("filename=%s format error" % str(filename))

def handle_parent(labelme_parent_dir, detect_dir):
    print("handle_parent() start")

    dir_names = os.listdir(labelme_parent_dir)
    print("handle_parent() labelme_parent_dir=%s,len(dir_names)=%d" % (labelme_parent_dir, len(dir_names)))

    for dir_name in dir_names:
        labelme_dir = os.path.join(labelme_parent_dir, dir_name)
        if os.path.isdir(labelme_dir) and not dir_name.startswith("__"):
            handle(labelme_dir=labelme_dir, detect_dir=detect_dir, flag=dir_name)


if __name__ == '__main__':
    print("__main__")

    handle(
        labelme_dir="E:\\data\\20240918cocos_bicycle",
        detect_dir="E:\\data\\20240918cocos_bicycle_detect\\train"
    )
    
    

    """ 

    handle_parent(
        labelme_parent_dir="D:\\datasets\\sample\\knife_gun_group",
        detect_dir="D:\\datasets\\bxc_detect_sample\\knife_gun_group20240828001\\train"
    )
    """
