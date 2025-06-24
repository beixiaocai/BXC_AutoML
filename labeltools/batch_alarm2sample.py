import os
import argparse
import shutil

class Alarm2Sample():
    def __init__(self,alarm_dir,output_dir):
        self.alarm_dir = alarm_dir # D:\file\storage\alarm
        self.output_dir = output_dir

    def run(self):
        pass

        control_dir_names = os.listdir(self.alarm_dir)
        for control_dir_name in control_dir_names:
            if control_dir_name.startswith('control'):
                control_dir_path = os.path.join(self.alarm_dir, control_dir_name) # D:\file\storage\alarm\control5245985c64
                ymd_dir_names = os.listdir(control_dir_path)
                for ymd_dir_name in ymd_dir_names:
                    if len(ymd_dir_name) == 8:
                        ymd_dir_path = os.path.join(control_dir_path,ymd_dir_name) # D:\file\storage\alarm\control5245985c64\20240911
                        hms_dir_names = os.listdir(ymd_dir_path)
                        for hms_dir_name in hms_dir_names:
                            hms_dir_path = os.path.join(ymd_dir_path,hms_dir_name)# D:\file\storage\alarm\control5245985c64\20240911\222123
                            type_dir_names = os.listdir(hms_dir_path)

                            if len(type_dir_names) == 2:
                                type0_dir_name = str(type_dir_names[0]) # 0
                                type1_dir_name = str(type_dir_names[1]) # 1
                                if type0_dir_name == "0" and type1_dir_name == "1":
                                    type0_dir_path = os.path.join(hms_dir_path,type0_dir_name)# D:\file\storage\alarm\control5245985c64\20240911\222123\0
                                    type0_names = os.listdir(type0_dir_path)
                                    print("--------------------------------")
                                    for type0_name in type0_names:
                                        if type0_name.endswith('.jpg'):
                                            print(type0_name)

                                            type0_path = os.path.join(type0_dir_path,type0_name)


                                            if not os.path.exists(self.output_dir):
                                                os.makedirs(self.output_dir)

                                            output_name = control_dir_name + '_' + type0_name
                                            type0_output_path = os.path.join(self.output_dir,output_name)
                                            if os.path.exists(type0_output_path):
                                                os.remove(type0_output_path)

                                            print("src",type0_path)
                                            print("dst",type0_output_path)
                                            shutil.copy(type0_path,type0_output_path)








if __name__ == '__main__':
    alarm_dir = "D:\\file\\storage\\alarm"
    output_dir = "E:\\data\\alarm_sample20240911"
    handle = Alarm2Sample(alarm_dir=alarm_dir,output_dir=output_dir)
    handle.run()