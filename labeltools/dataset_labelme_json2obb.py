import json
import os
import time
import shutil

"""
yolo8或yolo11的obb数据集格式： 
class_index, x, y, x, y, x, y, x, y 就是类别码和四个坐标

！！注意YOLO的数据集格式的坐标都是要归一化的！！！
"""

g_label2index = {
    "cartons": 0,
    "ConveyorBelt": 1,
    "cart": 2
}

print("g_label2index:", len(g_label2index), g_label2index)

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
                print("parse1 success filename=%s,name=%s,len(names)=%d " % (filename, name, len(names)))
            else:
                if filename.endswith(".json"):
                    name = filename[0:-5]
                    print("parse2 success filename=%s,name=%s,len(names)=%d " % (filename, name, len(names)))

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
                            if shape_type == "polygon":
                                # 多边形
                                if len(points)  == 4:
                                    x1 = float(points[0][0])
                                    y1 = float(points[0][1])
                                    x2 = float(points[1][0])
                                    y2 = float(points[1][1])
                                    x3 = float(points[2][0])
                                    y3 = float(points[2][1])
                                    x4 = float(points[3][0])
                                    y4 = float(points[3][1])

                                    if x1 > 0 and x2 < imageWidth:

                                        x1_f = x1 / float(imageWidth)
                                        y1_f = y1 / float(imageHeight)
                                        x2_f = x2 / float(imageWidth)
                                        y2_f = y2 / float(imageHeight)
                                        x3_f = x3 / float(imageWidth)
                                        y3_f = y3 / float(imageHeight)
                                        x4_f = x4 / float(imageWidth)
                                        y4_f = y4 / float(imageHeight)

                                        label_index = g_label2index.get(label, None)
                                        if label_index is None:
                                            print("\t未定义的标签名,json_filepath=%s" % json_filepath)
                                        else:
                                            line_content = "%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n" % (label_index,
                                                                                                             x1_f, y1_f,x2_f, y2_f, x3_f, y3_f, x4_f, y4_f, )
                                            save_label_f.write(line_content)
                                            success_count += 1
                                    else:
                                        print("发生错误,imagePath_abs=%s,e=%s" % (imagePath_abs, str("目标框超过了背景范围")))

                                else:
                                    print("发生错误,imagePath_abs=%s,e=%s" % (imagePath_abs, str("多边形边数不对")))
                            else:
                                print("发生错误,imagePath_abs=%s,e=%s" % (imagePath_abs, str("图形类别不对")))
                            j += 1

                        save_label_f.close()

                    except Exception as e:
                        print("发生未知错误,imagePath_abs=%s,e=%s"%(imagePath_abs, str(e)))

                    if success_count > 0:
                        shutil.copyfile(imagePath_abs, save_image_filepath)
                    else:
                        print("发生错误,imagePath_abs=%s,e=%s" % (imagePath_abs, str("无目标，已删除label文件")))
                        try:
                            os.remove(save_label_filepath)
                        except:
                            pass

                # except Exception as e:
                #     print("\t报错：第%d张图片%s" % (index, filename), e)

                index += 1
            else:
                print("parse error filename=%s,len(names)=%d " % (filename, len(names)))

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
    # handle(
    #     labelme_dir="D:\\file\\data",
    #     detect_dir="F:\\ai\\data\\20250712factory\\factory_group0717_detect_obb\\train"
    # )
    

    handle_parent(
        labelme_parent_dir="F:\\ai\\data\\20250712factory\\factory_group0717",
        detect_dir="F:\\ai\\data\\20250712factory\\factory_group0717_detect_obb\\train"
    )

