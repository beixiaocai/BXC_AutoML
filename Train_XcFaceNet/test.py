from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from nets.facenet import Facenet
from utils.utils import preprocess_input, resize_image, show_config

class Test(object):

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self,model_path,input_shape,backbone,letterbox_image,cuda):
        self.model_path = model_path
        self.input_shape = input_shape
        self.backbone = backbone
        self.letterbox_image = letterbox_image
        self.cuda=cuda
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.net = Facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]],
                                   letterbox_image=self.letterbox_image)
            image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]],
                                   letterbox_image=self.letterbox_image)

            image_1_tp = np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1))


            photo_1 = torch.from_numpy(
                np.expand_dims(image_1_tp, 0))
            photo_2 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))



            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)

        """
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()


        """

        return l1

if __name__ == "__main__":
    __params = {
        "model_path" : "runs/model.pth",
        "input_shape": [160, 160, 3],
        "backbone": "mobilenet",
        "letterbox_image": True,
        "cuda": False,
    }

    test = Test(**__params)
    url1 = "data/Adrien_Brody_0001.jpg"
    url2 = "data/Adrien_Brody_0012.jpg"

    print("url1=%s"%url1)
    print("url2=%s"%url2)

    image_1 = Image.open(url1)
    image_2 = Image.open(url2)

    t1 = time.time()
    probability = test.detect_image(image_1, image_2)
    t2 = time.time()

    print(t2 - t1, "两张图片的空间距离：", probability)
