from __future__ import print_function, division
import os
import sys
import subprocess


def class_process(dir_path, dst_dir_path, dir_name):
    class_path = os.path.join(dir_path, dir_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, dir_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    for file_name in os.listdir(class_path):
        # if '.avi' not in file_name:
        #   continue
        name, ext = os.path.splitext(file_name)
        dst_directory_path = os.path.join(dst_class_path, name)
        video_file_path = os.path.join(class_path, file_name)
        if not os.path.exists(dst_directory_path):
            os.makedirs(dst_directory_path)

        cmd = 'ffmpeg -i \"{}\" -vf scale=-1:240 \"{}/image_%05d.jpg\"'.format(video_file_path, dst_directory_path)
        print(cmd)
        subprocess.call(cmd, shell=True)
        print('\n')

def class_process_write_frames(dir_path, dir_name):
  class_path = os.path.join(dir_path, dir_name)
  if not os.path.isdir(class_path):
    return

  for file_name in os.listdir(class_path):
    video_dir_path = os.path.join(class_path, file_name)
    image_indices = []
    for image_file_name in os.listdir(video_dir_path):
      if 'image' not in image_file_name:
        continue
      image_indices.append(int(image_file_name[6:11]))

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True)
      n_frames = image_indices[0]
      print(video_dir_path, n_frames)
    with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
      dst_file.write(str(n_frames))

if __name__ == "__main__":

    ucf_dir = 'E:\\ai\\datasets\\UCF-101'
    ucf2image_dir = 'E:\\ai\\datasets\\UCF-101_cnn_lstm\\image_data'

    if not os.path.exists(ucf2image_dir):
        os.makedirs(ucf2image_dir)

    for dir_name in os.listdir(ucf_dir):
        class_process(ucf_dir, ucf2image_dir, dir_name)

    for dir_name in os.listdir(ucf2image_dir):
        class_process_write_frames(ucf2image_dir, dir_name)

