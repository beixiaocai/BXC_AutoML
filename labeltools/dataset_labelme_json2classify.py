import json
import os
import cv2
import time


def handle(labelme_dir, classify_dir):
    filenames = os.listdir(labelme_dir)
    print("handle() labelme_dir=%s,len(filenames)=%d" % (labelme_dir, len(filenames)))

    flag = str(int(time.time()))
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
                try:
                    json_filepath = os.path.join(labelme_dir, filename)
                    f = open(json_filepath, "r")
                    content = f.read()
                    f.close()

                    json_data = json.loads(content)
                    # version = json_data.get("version")
                    shapes = json_data.get("shapes")
                    imagePath = json_data.get("imagePath")
                    image_filepath = os.path.join(labelme_dir, imagePath)
                    if os.path.exists(image_filepath):
                        image = cv2.imread(image_filepath)
                        j = 0
                        for shape in shapes:
                            label = shape.get("label")
                            shape_type = shape.get("shape_type")
                            points = shape.get("points")
                            # print(label, shape_type, points)

                            x1 = int(points[0][0])
                            y1 = int(points[0][1])
                            x2 = int(points[1][0])
                            y2 = int(points[1][1])

                            
                            obj = image[y1:y2, x1:x2]
                            try:
                                obj_output_dir = os.path.join(classify_dir, label)
                                if not os.path.exists(obj_output_dir):
                                    os.makedirs(obj_output_dir)
                                obj_output_filepath = os.path.join(obj_output_dir, "%s-%s-%d.jpg" % (flag, name, j))
                                #print(json_filepath,x1,y1,x2,y2,obj)
                                #print(obj_output_filepath)
                                #print(obj.shape)
                               
                                cv2.imwrite(obj_output_filepath, obj)
                            except Exception as e:
                                pass
                            j += 1

                except Exception as e:
                    print("\t报错：第%d张图片%s" % (index, filename),"报错原因：", e)

                index += 1
            else:
                print("filename=%s format error" % str(filename))


def handle_parent(labelme_parent_dir, classify_dir):
    dir_names = os.listdir(labelme_parent_dir)
    print("handle_parent() labelme_parent_dir=%s,len(dir_names)=%d" % (labelme_parent_dir, len(dir_names)))
    for dir_name in dir_names:
        labelme_dir = os.path.join(labelme_parent_dir, dir_name)
        if os.path.isdir(labelme_dir) and not dir_name.startswith("__"):
            handle(labelme_dir=labelme_dir,classify_dir=classify_dir)


if __name__ == '__main__':
    handle_parent(labelme_parent_dir="D:\\datasets\\sample\\knife-gun_group1",
                  classify_dir="D:\\datasets\\bxc_classify_sample\\knife-gun_group120240828001")

    """


    handle(
        labelme_dir="D:\\datasets\\sample\\fight_group2",
        classify_dir="D:\datasets\\bxc_classify_sample\\fight_group0811002"
    )
    
    handle(
        labelme_dir="E:\\ai\\video\\bi2tYjJGAUM",
        classify_dir="E:\\ai\\video\\bi2tYjJGAUM_classify0807"
    )
    handle(
        labelme_dir="E:\\ai\\video\\fight1",
        classify_dir="E:\\ai\\video\\fight1_classify"
    )
    handle(
        labelme_dir="E:\\ai\\video\\fight2",
        classify_dir="E:\\ai\\video\\fight2_classify"
    )
    
    """
