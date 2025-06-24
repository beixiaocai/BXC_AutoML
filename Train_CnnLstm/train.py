import torch
import torch.nn as nn
import tensorboardX
import os
import random
import numpy as np
from utils import AverageMeter, calculate_accuracy
from torch.utils.data import DataLoader
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def train_epoch(model, data_loader, criterion, optimizer, epoch, log_interval, device):
    model.train()

    train_loss = 0.0
    losses = AverageMeter()
    accuracies = AverageMeter()
    for batch_idx, (data, targets) in enumerate(data_loader):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        train_loss += loss.item()
        losses.update(loss.item(), data.size(0))
        accuracies.update(acc, data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / log_interval
            print('train_epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(data_loader.dataset), 100. * (batch_idx + 1) / len(data_loader),
                avg_loss))
            train_loss = 0.0

    print('train_epoch end ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(
        len(data_loader.dataset), losses.avg, accuracies.avg * 100))

    return losses.avg, accuracies.avg


def val_epoch(model, data_loader, criterion, device):
    model.eval()

    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for (data, targets) in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)

            loss = criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)

            losses.update(loss.item(), data.size(0))
            accuracies.update(acc, data.size(0))

    # show info
    print('val_epoch end  ({:d} samples): Average loss: {:.4f}\tAcc: {:.4f}%'.format(len(data_loader.dataset), losses.avg,
                                                                                   accuracies.avg * 100))
    return losses.avg, accuracies.avg


def get_loaders(opt):
    """ Make dataloaders for train and validation sets
	"""
    # train loader
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    spatial_transform = Compose([
        # crop_method,
        Scale((opt.sample_size, opt.sample_size)),
        # RandomHorizontalFlip(),
        ToTensor(opt.norm_value), norm_method
    ])
    temporal_transform = TemporalRandomCrop(16)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform,temporal_transform, target_transform)
    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True)

    # validation loader
    spatial_transform = Compose([
        Scale((opt.sample_size, opt.sample_size)),
        # CenterCrop(opt.sample_size),
        ToTensor(opt.norm_value), norm_method
    ])
    target_transform = ClassLabel()
    temporal_transform = LoopPadding(16)
    validation_data = get_validation_set(
        opt, spatial_transform, temporal_transform, target_transform)
    val_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True)
    return train_loader, val_loader


def train():
    opt = parse_opts()
    print(opt)
    model_save_dir = opt.model_save_dir

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA for PyTorch
    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

    # tensorboard
    summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

    # defining model
    model = generate_model(opt, device)
    # get data loaders
    train_loader, val_loader = get_loaders(opt)

    # optimizer
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

    # scheduler = lr_scheduler.ReduceLROnPlateau(
    # 	optimizer, 'min', patience=opt.lr_patience)
    criterion = nn.CrossEntropyLoss()

    # resume model
    if opt.resume_path:
        print("")
        checkpoint = torch.load(opt.resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("迭代训练模型，重新迭代起始周期 {}".format(checkpoint['epoch']))
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 1

    # start training
    for epoch in range(start_epoch, opt.n_epochs + 1):
        print(f"@Train epoch={epoch} start")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)

        val_loss, val_acc = val_epoch(
            model, val_loader, criterion, device)

        # saving weights to checkpoint
        if epoch % opt.save_interval == 0:
            # scheduler.step(val_loss)
            # write summary
            summary_writer.add_scalar(
                'losses/train_loss', train_loss, global_step=epoch)
            summary_writer.add_scalar(
                'losses/val_loss', val_loss, global_step=epoch)
            summary_writer.add_scalar(
                'acc/train_acc', train_acc * 100, global_step=epoch)
            summary_writer.add_scalar(
                'acc/val_acc', val_acc * 100, global_step=epoch)

            model_state = {'epoch': epoch,
                           'state_dict': model.state_dict(),
                           'optimizer_state_dict': optimizer.state_dict()}

            model_name = f'{opt.model}-epoch-{epoch}-acc-{val_acc}-loss-{val_loss}.pth'
            model_filepath = os.path.join(model_save_dir, model_name)
            torch.save(model_state, model_filepath)

            print("@Epoch={epoch} save model, name={model_name}\n".format(epoch=epoch,
                                                                          model_name=model_name
                                                                          ))


if __name__ == "__main__":
    train()
