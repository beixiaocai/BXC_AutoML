import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.facenet import Facenet
from nets.facenet_training import (get_lr_scheduler, set_optimizer_lr,
                                   triplet_loss, weights_init)
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, dataset_collate
from utils.utils import (get_num_classes, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":

    is_use_cuda     = True if torch.cuda.is_available() else False
    seed            = 11
    distributed     = False # 用于指定是否使用单机多卡分布式运行
    sync_bn         = False
    fp16            = False
    train_desc_path = "datasets/train.txt" # 训练数据集描述文件
    lfw_path    = "datasets/lfw" # LFW评估数据集的文件路径和对应的txt文件
    lfw_desc_path  = "datasets/lfw.txt" # LFW评估数据集的文件路径和对应的txt文件
    lfw_eval_flag = True  # 是否开启LFW评估

    input_shape     = [160, 160, 3] # 输入图像大小，常用设置如[112, 112, 3]
    backbone        = "mobilenet"
    pretrained      = True # 是否启用预训练模型
    pre_model_path      = "pre_facenet_mobilenet.pth" # 预训练模型
    batch_size      = 66
    Init_Epoch      = 0
    Epoch           = 100
    Init_lr             = 1e-3 # 模型的初始化学习率
    Min_lr              = Init_lr * 0.01 # 模型的最小学习率，默认为最大学习率的0.01
    optimizer_type      = "adam" # 优化器 当使用Adam优化器时建议设置  Init_lr=1e-3，当使用SGD优化器时建议设置   Init_lr=1e-2
    momentum            = 0.9 # 优化器内部使用到的momentum参数
    weight_decay        = 0 # 权值衰减，可防止过拟合
    lr_decay_type       = "cos"
    save_period         = 1 # 多少个epoch保存一次权值，默认每个世代都保存
    save_dir            = 'runs' # 权值与日志文件保存的文件夹

    num_workers     = 4 # 用于设置是否使用多线程读取数据，内存较小的电脑可以设置为2或者0
    seed_everything(seed)
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    num_classes = get_num_classes(train_desc_path)
    #---------------------------------#
    #   载入模型并加载预训练权重
    #---------------------------------#
    model = Facenet(backbone=backbone, num_classes=num_classes, pretrained=pretrained)
    print('is_use_cuda: {}.'.format(is_use_cuda))
    print('device: {}.'.format(device))
    if pretrained:
        print('pre_model_path: {}.'.format(pre_model_path))

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(pre_model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        #------------------------------------------------------#
        #   显示没有匹配上的Key
        #------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    loss            = triplet_loss()
    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if is_use_cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    #---------------------------------#
    #   LFW估计
    #---------------------------------#
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_path, pairs_path=lfw_desc_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None

    #-------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #-------------------------------------------------------#
    val_split = 0.01
    with open(train_desc_path,"r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    show_config(
        num_classes = num_classes, backbone = backbone, model_path = pre_model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Epoch = Epoch, batch_size = batch_size, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
    )

    if True:
        if batch_size % 3 != 0:
            raise ValueError("batch_size must be the multiple of 3.")
        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 64
        lr_limit_max    = 1e-3 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        #---------------------------------------#
        #   构建数据集加载器。
        #---------------------------------------#
        train_dataset   = FacenetDataset(input_shape, lines[:num_train], num_classes, random = True)
        val_dataset     = FacenetDataset(input_shape, lines[num_train:], num_classes, random = False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True
        
        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size//3, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=val_sampler, 
                                worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
                
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, is_use_cuda, LFW_loader, batch_size//3, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
