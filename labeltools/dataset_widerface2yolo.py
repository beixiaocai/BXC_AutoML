import os
import shutil
from datetime import datetime
import cv2

def convert_wider_to_yolo(wider_annotations_file, wider_images_dir,save_dir, class_id=0):
    # 创建输出目录
    detect_images_dir = os.path.join(save_dir, "images")
    detect_labels_dir = os.path.join(save_dir, "labels")
    if not os.path.exists(detect_images_dir):
        os.makedirs(detect_images_dir)
    if not os.path.exists(detect_labels_dir):
        os.makedirs(detect_labels_dir)


    # 读取WIDER FACE标注文件
    with open(wider_annotations_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        # 读取图像路径
        image_path = lines[i].strip()
        i += 1

        # 读取人脸框数量
        # print("lines[i]:",i,type(lines[i]),lines[i])
        num_faces = int(lines[i].strip())
        i += 1

        if num_faces > 0: # 当num_face==0时，后面还紧跟着一行 0 0 0 0 0 0 0 0 0 0
            # 读取每个人脸框的坐标
            bboxes = []
            for _ in range(num_faces):
                bbox = list(map(int, lines[i].strip().split()[:4]))
                bboxes.append(bbox)
                i += 1

            # 生成YOLO格式的标注文件
            image_name = os.path.basename(image_path)
            save_name = os.path.splitext(image_name)[0] # 9_Press_Conference_Press_Conference_9_266

            org_image_filepath = os.path.join(wider_images_dir, image_path) # /xxx/9--Press_Conference/9_Press_Conference_Press_Conference_9_131.jpg
            if os.path.exists(org_image_filepath):

                try:
                    org_image = cv2.imread(org_image_filepath)
                    h,w,c = org_image.shape

                    save_image_filepath = os.path.join(detect_images_dir, save_name+".jpg")
                    save_label_filepath = os.path.join(detect_labels_dir, save_name+".txt")

                    with open(save_label_filepath, 'w') as f:
                        for bbox in bboxes:
                            x_min, y_min, width, height = bbox
                            x_center = (x_min + width / 2) / 1.0
                            y_center = (y_min + height / 2) / 1.0
                            norm_width = width / 1.0
                            norm_height = height / 1.0

                            x_center = round(x_center/w,6)
                            y_center = round(y_center/h,6)

                            norm_width = round(norm_width/w,6)
                            norm_height = round(norm_height/h,6)

                            # 写入YOLO格式的标注
                            f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")

                    print("Done",i,save_name,image_path)


                    shutil.copyfile(org_image_filepath, save_image_filepath)
                except Exception as e:
                    print("handle image error:",e,org_image_filepath)


        else:
            i += 1 # 当num_face==0时，后面还紧跟着一行 0 0 0 0 0 0 0 0 0 0

if __name__ == '__main__':
    print(datetime.now(),"开始处理训练集")
    wider_annotations_file = 'F:\\ai\\data\\face_group\\wider_face_split\\wider_face_train_bbx_gt.txt'
    wider_images_dir = 'F:\\ai\\data\\face_group\\WIDER_train\\images'
    save_dir = 'F:\\ai\\data\\face_group\\detect\\train'

    convert_wider_to_yolo(wider_annotations_file, wider_images_dir, save_dir)

    print(datetime.now(),"开始处理验证集")
    wider_annotations_file = 'F:\\ai\\data\\face_group\\wider_face_split\\wider_face_val_bbx_gt.txt'
    wider_images_dir = 'F:\\ai\\data\\face_group\\WIDER_val\\images'
    save_dir = 'F:\\ai\\data\\face_group\\detect\\valid'

    convert_wider_to_yolo(wider_annotations_file, wider_images_dir, save_dir)