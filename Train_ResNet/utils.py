import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import requests
from io import StringIO
from urllib import parse
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def CreateLogger(logDir,prefix,is_show_console=False):
    if not os.path.exists(logDir):
        os.makedirs(logDir)

    LOGGER_TIMEFMT = "%Y%m%d-%H%M%S"
    # LOGGER_WHEN = 'd'
    # LOGFILE_BACKUPCOUNT = 7

    fileName = os.path.join(logDir, prefix+"-%s.log.txt" % (datetime.now().strftime(LOGGER_TIMEFMT)))
    level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(level)
    # formatter = logging.Formatter('%(asctime)s %(name)s %(lineno)s VideoAnalyzer [%(levelname)s] %(message)s')
    formatter = logging.Formatter('%(asctime)s %(lineno)s [%(levelname)s] %(message)s')

    # 最基础
    fileHandler = logging.FileHandler(fileName, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
    fileHandler.setLevel(level)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # 时间滚动切分
    # when:备份的时间单位，backupCount:备份保存的时间长度
    # timedRotatingFileHandler = TimedRotatingFileHandler(fileName,
    #                                                     when=LOGGER_WHEN,
    #                                                     backupCount=LOGFILE_BACKUPCOUNT,
    #                                                     encoding='utf-8')
    #
    # timedRotatingFileHandler.setLevel(level)
    # timedRotatingFileHandler.setFormatter(formatter)
    # logger.addHandler(timedRotatingFileHandler)

    # 控制台打印
    if is_show_console:
        streamHandler = logging.StreamHandler()
        streamHandler.setLevel(level)
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger

def image_loader_url(url, tsfms):
    # image = Image.open(url).convert('RGB')
    headers = {
        'User-Agent': 'BXC_AutoML',
        'Host': parse.urlparse(url).hostname,
        'Origin': parse.urlparse(url).hostname,
        'Connection': 'keep-alive',
        # 'Referer': urlparse.urlparse(url).hostname,
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    image = requests.get(url, timeout=3, headers=headers).content
    image = Image.open(StringIO(image)).convert('RGB')
    image_tensor = tsfms(image)
    # fake batch dimension required to fit network's input dimensions
    return image_tensor

def randomFlip(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def randomBlur(image, prob=0.5):
    rnd = np.random.random_sample()
    if rnd < prob:
        return image.filter(ImageFilter.BLUR)
    return image

def randomRotation(image, prob=0.5, angle=(1, 60)):
    rnd = np.random.random_sample()
    if rnd < prob:
        random_angle = np.random.randint(angle[0], angle[1])
        return image.rotate(random_angle)
    return image

def randomColor(image, prob=0.7, factor=(1, 90)):
    rnd = np.random.random_sample()
    if rnd < prob:
        # Factor 1.0 always returns a copy of the original image,
        # lower factors mean less color (brightness, contrast, etc), and higher values more
        random_factor = np.random.randint(2, 18) / 10.
        color_image = ImageEnhance.Color(image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = np.random.randint(5, 18) / 10.
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = np.random.randint(2, 18) / 10.
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
    return image

def randomGaussian(image, prob=0.5, mean=0, sigma=10):
    rnd = np.random.random_sample()
    if rnd < prob:
        img_array = np.asarray(image)
        noisy_img = img_array + np.random.normal(mean, sigma, img_array.shape)
        noisy_img = np.clip(noisy_img, 0, 255)

        return Image.fromarray(np.uint8(noisy_img))
    return image
