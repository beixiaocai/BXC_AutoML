import os
import time
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from PIL import Image
import argparse
from utils import CreateLogger
logger = CreateLogger(logDir="log",prefix="test",is_show_console=True)
MODEL_NAME = "resnet50"

def run():
    logger.info("Train_ResNet 开始测试")
    logger.info(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize, ])

    checkpoint = torch.load(args.checkpoint)
    logger.info("checkpoint.keys()=%s" % str(checkpoint.keys()))
    classes = checkpoint['classes']
    logger.info("checkpoint.classes=%s" % str(classes))
    logger.info("checkpoint.epoch=%s" % str(checkpoint["epoch"]))

    model = torchvision.models.__dict__[MODEL_NAME](pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))

    device = args.device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=args.device)
    # model.cuda()
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for filename in os.listdir(args.test_dir):

        try:
            t1 = time.time()
            image_filepath = os.path.join(args.test_dir, filename)

            # image_tensor = transformation(Image.open(image_)).float()
            image_tensor = transformation(Image.open(image_filepath).convert('RGB')).float()
            image_tensor = image_tensor.unsqueeze_(0)
            if device == "cpu":
                output = model(image_tensor)
            else:
                output = model(image_tensor.cuda())

            index = output.data.cpu().numpy().argmax()
            # print(output.data.cpu().numpy())
            label = classes[index]
            t2 = time.time()
            t_spend = (t2 - t1) * 1000
            logger.info('推理结果: {}->{}->{} 耗时{:.3f}毫秒'.format(filename, classes[index], index, t_spend))
        except Exception as e:
            logger.error('推理失败: {}->{}'.format(filename, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train_ResNet Test')
    parser.add_argument('--test-dir', default='dataset/test', help='dataset')
    parser.add_argument('--device', default="cpu", help='device')
    parser.add_argument('--checkpoint', default='./checkpoint/checkpoint.pth', help='checkpoint')

    args = parser.parse_args()

    run()
