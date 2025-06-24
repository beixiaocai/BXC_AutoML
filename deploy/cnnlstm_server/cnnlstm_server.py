import torch
import torch.nn.functional as F
from mean import get_mean, get_std

from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
import json
import cv2
import os
from PIL import Image
from datetime import datetime
from cnnlstm import CNNLSTM
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(__file__)
print("BASE_DIR=%s" % BASE_DIR)

app = Flask(__name__)

def resume_model(opt, model):
    """ Resume model 
    """
    checkpoint = torch.load(opt.resume_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])


def predict(clip, model):
    norm_value = 1
    mean_dataset = "activitynet"
    no_mean_norm = False
    std_norm = False
    mean = get_mean(norm_value, dataset=mean_dataset)
    std = get_std(norm_value)

    if no_mean_norm and not std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not std_norm:
        norm_method = Normalize(mean, [1, 1, 1])
    else:
        norm_method = Normalize(mean, std)

    spatial_transform = Compose([
        Scale((150, 150)),
        # Scale(int(opt.sample_size / opt.scale_in_test)),
        # CornerCrop(opt.sample_size, opt.crop_position_in_test),
        ToTensor(norm_value), norm_method
    ])
    if spatial_transform is not None:
        # spatial_transform.randomize_parameters()
        clip = [spatial_transform(img) for img in clip]

    clip = torch.stack(clip, dim=0)
    clip = clip.unsqueeze(0)
    with torch.no_grad():
        print(clip.shape)
        outputs = model(clip)
        outputs = F.softmax(outputs)
    print(outputs)
    scores, idx = torch.topk(outputs, k=1)
    index = scores > 0
    scores_result = scores[index]
    idx_result = idx[index]

    class_index = idx_result.item()
    class_score = scores_result.item()
    class_score = float("%.4f" % class_score)

    return class_index, class_score


@app.route('/', methods=['POST', 'GET'])
def index():
    return "cnnlstm_server"


@app.route('/algorithm', methods=['POST', 'GET'])
def algorithm():
    ret = False
    msg = "未知错误"
    info = {}
    try:
        request_params = request.get_json()
        print("request_params=", datetime.now(), request_params)


        image_dir = "D:\\Project\\BXC_VideoAnalyzer_v3\\data\\cnnlstmdata"
        # filenames = os.listdir(image_dir)

        clip = []
        for i in range(16):
            filepath = os.path.join(image_dir, "%d.jpg" % i)
            if os.path.exists(filepath):
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # img = Image.fromarray(img.astype('uint8'), 'RGB')
                image = Image.fromarray(image)
                clip.append(image)
            else:
                raise Exception("图片文件不存在:%s，推理异常" % filepath)

        if len(clip) == 16:
            class_index, class_score = predict(clip, model)
            class_name = class_names[class_index]
            print("预测输出结果:%s,得分:%.4f" % (class_name, class_score))

            info["class_index"] = class_index
            info["class_name"] = class_name
            info["class_score"] = class_score

            ret = True
            msg = "success"
        else:
            msg = "需要16张图片，当前仅%d张，推理异常" % len(clip)
    except Exception as e:
        msg = str(e)

    response_data = {
        "code": 1000 if ret else 0,
        "msg": msg,
        "info": info
    }
    return jsonify(response_data)


def test():
    test_url = "D:\\file\\video\\test13.mp4"
    # test_url = 0

    cam = cv2.VideoCapture(test_url)
    clip = []
    frame_count = 0
    while True:
        ret, img = cam.read()
        if ret:
            if frame_count == 16:
                class_index, class_score = predict(clip, model)
                print("预测输出结果:%s,得分:%.4f" % (class_names[class_index], class_score))

                frame_count = 0
                clip = []

            # img = cv2.resize(img, (224,224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = Image.fromarray(img.astype('uint8'), 'RGB')
            img = Image.fromarray(img)
            clip.append(img)
            frame_count += 1
        else:
            print("读取视频结束")
            break


if __name__ == "__main__":
    device = torch.device("cpu")
    model_path = "models/model100.pth"
    class_names = ["ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
                   "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress"]

    model = CNNLSTM(num_classes=len(class_names))
    model.to(device)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    # test()
    app.run(host="0.0.0.0", port=9710, debug=True)
