import cv2
import time
import os


def handle():
    filenames = os.listdir(images_dir)
    print(len(filenames))

    flag = str(int(time.time()))

    index = 0
    for filename in filenames:
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
            try:
                label_filename = name + ".txt"
                image_filepath = os.path.join(images_dir, filename)
                label_filepath = os.path.join(labels_dir, label_filename)

                image = cv2.imread(image_filepath)
                image_shape = image.shape

                image_height = image_shape[0]
                image_width = image_shape[1]

                f = open(label_filepath)
                lines = f.readlines()
                f.close()
                j = 0
                for line in lines:
                    line = line.strip().split("\n")[0]
                    line_v = line.split(" ")
                    if len(line_v) == 5:
                        class_index = int(line_v[0])
                        class_name = str(class_index)

                        x_center = int(float(line_v[1]) * image_width)
                        y_center = int(float(line_v[2]) * image_height)
                        w = int(float(line_v[3]) * image_width)
                        h = int(float(line_v[4]) * image_height)

                        x1 = int(x_center - w / 2)
                        x2 = int(x_center + w / 2)
                        y1 = int(y_center - h / 2)
                        y2 = int(y_center + h / 2)

                        if x1 > 0 and y1 > 0 and x2 < image_width and y2 < image_height and w > 10 and h > 10:

                            obj = image[y1:y2, x1:x2]

                            obj_output_dir = os.path.join(classify_dir, class_name)
                            if not os.path.exists(obj_output_dir):
                                os.makedirs(obj_output_dir)

                            obj_output_filepath = os.path.join(obj_output_dir, "%s-%s-%d.jpg" % (flag, name, j))
                            cv2.imwrite(obj_output_filepath, obj)

                            # cv2.imshow("obj-" + str(class_index), obj)
                            # cv2.waitKey(0)
                            j += 1

                index += 1
            except Exception as e:
                print("报错:", e)
        else:
            print("filename=%s format error" % str(filename))
            # if index > 3:
            #     break

    cv2.destroyAllWindows()


if __name__ == '__main__':

    images_dir = "F:\\ai\\data\\face_group\\detect\\train\\images"
    labels_dir = "F:\\ai\\data\\face_group\\detect\\train\\labels"
    classify_dir = "F:\\ai\\data\\face_group\\classify"

    handle()
