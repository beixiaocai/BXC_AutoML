import os

# class_list = ['apparatus', 'bend', 'cudgel', 'fall', 'fight', 'gun', 'knife', 'run', 'scissors', 'shield', 'sit', 'stand', 'stool']
class_list = ['apparatus', 'bend', 'cudgel', 'fall', 'fight', 'gun', 'knife', 'run', 'scissors', 'shield', 'sit', 'stand', 'stool']

class_labels = []
print("--------------转换前--------------")
print("len(class_list):",len(class_list))
print(class_list)

for class_label in class_list:
    class_labels.append(class_label)

print("--------------转换后--------------")
print("len(class_labels):",len(class_labels))
print(",".join(class_labels))