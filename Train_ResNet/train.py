import os
import time
import math
import argparse
import torch
import torch.utils.data
from torch import nn
import torchvision
from torchvision import transforms
from utils import CreateLogger

import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

"""
import torchvision

from torchvision.models.resnet import ResNet
from torchvision.models.resnet import ResNet34_Weights
from torchvision.models.resnet import ResNet18_Weights

from torchvision.models.vgg import VGG
from torchvision.models.vgg import VGG11_Weights
"""

logger = CreateLogger(logDir="log", prefix="train", is_show_console=True)

# MODEL_NAME = "vgg16"
# MODEL_NAME = "resnet101"
# MODEL_NAME = "resnet50"
MODEL_NAME = "resnet18"

def train_eval(model, criterion, val_loader, epoch):
    epoch_start = time.time()
    running_loss = 0.0
    running_corrects = 0
    val_data_len = len(val_loader.dataset)

    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            batch_start = time.time()
            if args.device == "cpu":
                pass
            else:
                image, target = image.cuda(), target.cuda()

            output = model(image)
            loss = criterion(output, target)

            _, preds = torch.max(output, 1)

            loss_ = loss.item() * image.size(0)  # this batch loss
            correct_ = torch.sum(preds == target.data)  # this batch correct number

            running_loss += loss_
            running_corrects += correct_

            batch_end = time.time()
            if i % args.print_freq == 0:
                pass
                # logger.info('[VAL] Epoch: {}/{}/{}, Batch: {}/{}, BatchAcc: {:.4f}, BatchLoss: {:.4f}, BatchTime: {:.4f}'.format(step,
                #       epoch, args.epochs, i, math.ceil(epoch_data_len/args.batch_size), correct_.double()/image.size(0),
                #       loss_/image.size(0), batch_end-batch_start))

    epoch_loss = running_loss / val_data_len
    epoch_acc = running_corrects.double() / val_data_len
    epoch_end = time.time()

    logger.info('[VAL@END] Epoch: {}/{}, acc: {:.4f}, loss: {:.4f}, time: {:.4f}s'.format(
        epoch, args.epochs, epoch_acc, epoch_loss, epoch_end - epoch_start))

    return epoch_acc


def train_epoch(model, criterion, optimizer, train_loader, epoch, val_loader, classes):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    data_len = len(train_loader.dataset)  # 总数居长度
    batch_len = math.ceil(data_len / args.batch_size)  # 当前周期内总训练批次

    logger.info("当前训练周期:{epoch},当前训练数据总数量:{data_len},当前训练数据总批次:{batch_len}".format(
        epoch=epoch,
        data_len=data_len,
        batch_len=batch_len
    ))

    for i, (image, target) in enumerate(train_loader):
        batch_start = time.time()
        if args.device == "cpu":
            pass
        else:
            image, target = image.cuda(), target.cuda()
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        _, preds = torch.max(output, 1)

        loss_ = loss.item() * image.size(0)  # this batch loss
        correct_ = torch.sum(preds == target.data)  # this batch correct number

        running_loss += loss_
        running_corrects += correct_

        batch_end = time.time()
        if i % args.print_freq == 0 and i > 0:
            logger.info(
                '[TRAIN] Epoch: {}/{}, {}/{}, acc: {:.4f}, loss: {:.4f}, time: {:.4f}'.format(
                    epoch, args.epochs,
                    i, batch_len,
                    correct_.double() / image.size(0),
                    loss_ / image.size(0),
                    batch_end - batch_start))

    # 验证start
    model.eval()  # 切换到验证模式
    val_acc = train_eval(model, criterion, val_loader, epoch)
    model.train()  # 恢复训练模式
    # 验证end

    lr = optimizer.param_groups[0]["lr"]
    epoch_loss = running_loss / data_len
    epoch_acc = running_corrects.double() / data_len
    epoch_end = time.time()
    logger.info('[TRAIN@END] Epoch: {}/{}, acc: {:.4f}, loss: {:.4f}, EpochTime: {:.4f} s, lr: {}'.format(epoch,
                                                                                                          args.epochs,
                                                                                                          epoch_acc,
                                                                                                          epoch_loss,
                                                                                                          epoch_end - epoch_start,
                                                                                                          lr))

    return epoch_loss, epoch_acc


def run():
    logger.info("Train_ResNet 开始训练")
    logger.info(args)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 加载训练样本集
    train_dir = os.path.join(args.data_dir, 'train')
    train_set = torchvision.datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(45),  # 随机旋转，-45度->45度之间随机
            # transforms.RandomResizedCrop(224),
            transforms.RandomCrop(224),
            # transforms.Grayscale(num_output_channels=3),
            # transforms.RandomAffine(45, shear=0.2),
            # transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),  # 亮度，对比度，饱和度，色度
            # transforms.RandomGrayscale(p=0.4), # 随机单通道覆盖三通道
            transforms.RandomHorizontalFlip(),
            # transforms.Lambda(utils.randomColor),
            # transforms.Lambda(utils.randomBlur),
            # transforms.Lambda(utils.randomGaussian),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, pin_memory=True)

    # 加载验证样本集
    val_dir = os.path.join(args.data_dir, 'val')
    val_set = torchvision.datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.CenterCrop(299),
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize, ]))
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True)

    classes = train_loader.dataset.classes
    logger.info("train_loader.classes=%s" % str(classes))
    val_classes = val_loader.dataset.classes
    logger.info("val_classes.classes=%s" % str(val_classes))

    # model = torchvision.models.__dict__[MODEL_NAME](pretrained=True)
    model = torchvision.models.__dict__[MODEL_NAME](weights=True)
    # model = torchvision.models.vgg16(weights=True)

    # 设置迁移模型不学习参数
    # for param in model.parameters():
    #     param.requires_grad = False

    # resnet
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))

    # vgg
    # in_features = model.fc6.in_features
    # model.fc = nn.Linear(in_features, len(classes))

    # 支持多卡训练
    # model = nn.DataParallel(model, device_ids=args.device)
    # model.cuda()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(args.device)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 学习率衰减策略
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma) # 按照周期衰减
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 80], gamma=args.lr_gamma)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    epoch_start = 1
    if args.resume:
        resume = args.resume
        checkpoint = torch.load(resume, map_location='cpu')
        logger.info("resume=%s,checkpoint.keys()=%s" % (resume, str(checkpoint.keys())))

        epoch_start = int(checkpoint["epoch"]) + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    epoch_end = epoch_start + args.epochs
    logger.info("epoch_start=%d,epoch_end=%d" % (epoch_start, epoch_end))

    for epoch in list(range(epoch_start, epoch_end + 1)):
        epoch_loss, epoch_acc = train_epoch(model, criterion, optimizer, train_loader, epoch, val_loader, classes)

        model_pth_filepath = os.path.join(args.checkpoint, 'model_{}_loss{:.4f}_acc{:.4f}.pth'.
                                      format(epoch, epoch_loss, epoch_acc))
        model_pt_filepath = os.path.join(args.checkpoint, 'model_{}_loss{:.4f}_acc{:.4f}.pt'.
                                      format(epoch, epoch_loss, epoch_acc))

        torch.save(
            {
                'model': model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                'classes': classes,
                'args': args,
                "epoch": epoch
            },
            model_pth_filepath
        )
        torch.save(
            model,
            model_pt_filepath
        )
        lr_scheduler.step()  # 周期学习率衰减

    logger.info("Train_ResNet 训练结束")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train_ResNet Train')
    parser.add_argument('--data-dir', default='dataset', help='dataset')
    parser.add_argument('--device', default="cpu", help='device')
    parser.add_argument('-b', '--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--checkpoint', default='./checkpoint', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        os.mkdir(args.checkpoint)

    run()
