"""
验证数据集生成lfw.txt文件的脚本

"""
import os.path
import random
import os
def create_matched_result(lfw_dir, all_number):
    matched_result = set()
    # 获取所有数据名字
    names = os.listdir(lfw_dir)
    while len(matched_result) < all_number:
        for name in names:
            length = len(os.listdir(os.path.join(lfw_dir, name)))
            if length <= 1:
                continue
            else:
                # 构造两个随机数
                number_1, number_2 = sorted(random.sample(range(1, length + 1), 2))
                s = name + '\t' + str(number_1) + '\t' + str(number_2)
                # 没有必要做判断，set会自行处理
                matched_result.add(s)
                if len(matched_result) == all_number:
                    break
    return list(matched_result)
def create_unmatched_result(lfw_dir, all_number):
    unmatched_result = set()
    names = os.listdir(lfw_dir)
    while len(unmatched_result) < all_number:
        # 随机取两个名字
        name1, name2 = sorted(random.sample(names, 2))
        numbers = []
        for name in [name1, name2]:
            images = os.listdir(os.path.join(lfw_dir, name))
            # 随机取人脸
            number = random.sample(range(1, len(images) + 1), 1)[0]
            numbers.append(number)
        # 没有必要做判断，set会自行处理
        s = name1 + '\t' + str(numbers[0]) + '\t' + name2 + '\t' + str(numbers[1])
        unmatched_result.add(s)
    return list(unmatched_result)
if __name__ == '__main__':
    lfw_dir = '../datasets/lfw'
    lfw_desc_path = '../datasets/lfw.txt'
    
    repeat_count = 10 # 重复次数
    pair_number = 300  # 每次匹配数量
    all_number = repeat_count * pair_number# 总数量
    matched_result = create_matched_result(lfw_dir, all_number) # 构造同一张人脸的集合
    unmatched_result = create_unmatched_result(lfw_dir, all_number) # 构造不同人脸的集合

    with open(lfw_desc_path, 'w') as f:
        f.write('%d %d\n' % (repeat_count, pair_number))
        for i in range(repeat_count):
            for pair in sorted(matched_result[i * pair_number: (i + 1) * pair_number]):
                f.write(pair + '\n')
            for pair in sorted(unmatched_result[i * pair_number: (i + 1) * pair_number]):
                f.write(pair + '\n')
