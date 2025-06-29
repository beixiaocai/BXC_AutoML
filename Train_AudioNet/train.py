import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import get_dataloaders
from model import SnoreClassifier, SnoreClassifierWithAttention

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
    
    Returns:
        平均损失和准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='训练')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    验证模型
    
    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        平均损失、准确率、预测结果和真实标签
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='验证')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 收集预测结果和真实标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
    
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    绘制训练指标
    
    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率
        val_accs: 验证准确率
        save_dir: 保存目录
    """
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(val_accs, label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def plot_confusion_matrix(cm, class_names, save_dir):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_dir: 保存目录
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--data_dir', type=str, default="data", help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='模型保存目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--use_attention', action='store_true', help='使用注意力机制')
    parser.add_argument('--sample_rate', type=int, default=16000, help='音频采样率')
    parser.add_argument('--max_duration', type=float, default=5.0, help='最大音频长度（秒）')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        max_duration=args.max_duration,
        num_workers=args.num_workers
    )

    # 创建模型
    if args.use_attention:
        model = SnoreClassifierWithAttention(num_classes=2, pretrained=True)
        print('使用带注意力机制的ResNet18')
    else:
        model = SnoreClassifier(num_classes=2, pretrained=True)
        print('使用标准ResNet18')

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 训练循环
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 更新学习率
        scheduler.step(val_loss)

        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f'保存最佳模型，验证准确率: {val_acc:.2f}%')

        # 每个epoch保存一次
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'val_loss': val_loss,
        }, os.path.join(args.save_dir, "model_epoch%d_%.2f_%.2f.pt" % (epoch + 1, val_loss, val_acc)))

    # 绘制训练指标
    plot_metrics(train_losses, val_losses, train_accs, val_accs, args.save_dir)

    class_names = ['打鼾', '非打鼾'],

    # 计算并绘制最终的混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, args.save_dir)

    # 打印分类报告
    report = classification_report(all_labels, all_preds, target_names=['打鼾', '非打鼾'])
    print('\n分类报告:')
    print(report)

    # 保存分类报告
    with open(os.path.join(args.save_dir, 'train_report.txt'), 'w') as f:
        f.write(report)

    print(f'训练完成。最佳验证准确率: {best_val_acc:.2f}%')
