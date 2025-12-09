from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.S3DISDataLoader import S3DISDataset
import torch.nn.functional as F
import torch.nn as nn
import model as models
import numpy as np
from torch.utils.data import DataLoader
from util.util import to_categorical, compute_overall_iou, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
import sys
import time

# 自动获取路径，兼容 Linux/Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))

# S3DIS 类别
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}


def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)


def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))


def train(args, io):
    num_sem = 13  # S3DIS 13类
    device = torch.device("cuda" if args.cuda else "cpu")

    # 初始化模型，使用 pointMLP 函数
    # 注意：确保 model/__init__.py 中正确引用了 pointMLP.py
    if args.model == 'pointMLP':
        model = models.pointMLP.pointMLP(num_classes=num_sem).to(device)
    else:
        # 兼容其他模型调用方式
        try:
            model = models.__dict__[args.model](num_sem).to(device)
        except:
            print(f"Error: Model {args.model} not found.")
            return

    io.cprint(str(model))

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.resume:
        try:
            checkpoint_path = "checkpoints/%s/best_insiou_model.pth" % args.exp_name
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))['model']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k in state_dict.keys():
                name = k[7:] if k.startswith('module.') else k  # remove `module.`
                new_state_dict[name] = state_dict[k]
            model.load_state_dict(new_state_dict)
            print("Resume training model...")
        except:
            print("Resume failed, training from scratch...")
    else:
        print("Training from scratch...")

    # DataLoader
    train_data = S3DISDataset(split='train', data_root=args.seg_path, num_point=args.num_points,
                              test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    test_data = S3DISDataset(split='test', data_root=args.seg_path, num_point=args.num_points,
                             test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=True, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=True)

    # Optimizer
    if args.use_sgd:
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100)
    else:
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    best_acc = 0.0
    best_class_iou = 0.0
    best_class_acc = 0.0

    for epoch in range(args.epochs):
        train_epoch(train_loader, model, opt, scheduler, epoch, num_sem, io, args)
        test_metrics, total_per_cat_iou = test_epoch(test_loader, model, num_sem, epoch, io, args)

        # Save Best Models
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            save_checkpoint(model, opt, epoch, best_acc, 'best_acc_model', args)

        if test_metrics['class_avg_acc'] > best_class_acc:
            best_class_acc = test_metrics['class_avg_acc']
            save_checkpoint(model, opt, epoch, best_class_acc, 'best_class_avg_acc_model', args)

        if test_metrics['class_avg_iou'] > best_class_iou:
            best_class_iou = test_metrics['class_avg_iou']
            save_checkpoint(model, opt, epoch, best_class_iou, 'best_class_avg_iou_model', args)

            # Print per-class IoU
            io.cprint(f'Epoch {epoch + 1} Best Class Avg IoU: {best_class_iou:.5f}')
            for i, cat in enumerate(seg_label_to_cat.values()):
                io.cprint(f'{cat}: {total_per_cat_iou[i]:.5f}')

    io.cprint(f'Final Best Acc: {best_acc:.5f}')
    io.cprint(f'Final Best Class Avg IoU: {best_class_iou:.5f}')


def save_checkpoint(model, opt, epoch, metric, name, args):
    state = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'epoch': epoch,
        'metric': metric
    }
    torch.save(state, 'checkpoints/%s/%s.pth' % (args.exp_name, name))


def train_epoch(train_loader, model, opt, scheduler, epoch, num_sem, io, args):
    train_loss = 0.0
    count = 0.0
    total_correct = 0
    total_seen = 0

    model.train()

    for batch_id, (points, label) in tqdm(enumerate(train_loader), total=len(train_loader),
                                          desc=f"Train Epoch {epoch + 1}"):
        # points: [B, N, 9], label: [B, N]

        # 数据增强在 Tensor 上做可能比较慢，建议移到 DataLoader 中
        # 这里为了兼容原始逻辑保留，但简化了操作
        if args.cuda:
            points = points.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()
        else:
            points = points.float()
            label = label.long()

        # 修复：维度转置 [B, N, 9] -> [B, 9, N] 以适应 PointMLP 输入
        points = points.permute(0, 2, 1)

        opt.zero_grad()

        # Forward
        seg_pred = model(points)  # [B, N, 13]

        # Reshape for Loss
        seg_pred_view = seg_pred.contiguous().view(-1, num_sem)  # [B*N, 13]
        label_view = label.view(-1)  # [B*N]

        loss = F.nll_loss(seg_pred_view, label_view)
        loss.backward()
        opt.step()

        # Metrics
        pred_val = seg_pred_view.max(1)[1]
        correct = pred_val.eq(label_view).sum().item()
        total_correct += correct
        total_seen += len(label_view)
        train_loss += loss.item() * args.batch_size
        count += args.batch_size

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 1e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 1e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 1e-5

    accuracy = total_correct / total_seen
    io.cprint('Train %d, loss: %.5f, train acc: %.5f, lr: %.6f' % (
        epoch + 1, train_loss / count, accuracy, opt.param_groups[0]['lr']))


def test_epoch(test_loader, model, num_sem, epoch, io, args):
    metrics = defaultdict(lambda: 0.0)
    model.eval()

    total_correct = 0
    total_seen = 0
    total_seen_class = [0] * num_sem
    total_correct_class = [0] * num_sem
    total_iou_deno_class = [0] * num_sem
    test_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch_id, (points, label) in tqdm(enumerate(test_loader), total=len(test_loader),
                                              desc=f"Test Epoch {epoch + 1}"):
            if args.cuda:
                points = points.cuda(non_blocking=True).float()
                label = label.cuda(non_blocking=True).long()
            else:
                points = points.float()
                label = label.long()

            # 修复：维度转置
            points = points.permute(0, 2, 1)

            seg_pred = model(points)  # [B, N, 13]

            # Loss
            seg_pred_view = seg_pred.contiguous().view(-1, num_sem)
            label_view = label.view(-1)
            loss = F.nll_loss(seg_pred_view, label_view)
            test_loss += loss.item() * args.batch_size
            count += args.batch_size

            # Metrics
            pred_val = seg_pred.contiguous().view(-1, num_sem).max(1)[1].cpu().numpy()
            target_val = label.view(-1).cpu().numpy()

            total_correct += np.sum(pred_val == target_val)
            total_seen += len(target_val)

            for l in range(num_sem):
                total_seen_class[l] += np.sum(target_val == l)
                total_correct_class[l] += np.sum((pred_val == l) & (target_val == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (target_val == l))

    mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
    mAcc_cls = np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))
    acc = total_correct / total_seen
    final_per_cat_iou = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)

    metrics['accuracy'] = acc
    metrics['class_avg_iou'] = mIoU
    metrics['class_avg_acc'] = mAcc_cls

    io.cprint('Test %d, loss: %.5f, acc: %.5f, mIoU: %.5f, mAcc: %.5f' % (
        epoch + 1, test_loss / count, acc, mIoU, mAcc_cls))

    return metrics, final_per_cat_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Semantic Segmentation')
    # 修复：默认模型名称改为 pointMLP
    parser.add_argument('--model', type=str, default='pointMLP', help='Model name')
    parser.add_argument('--exp_name', type=str, default='demo1', help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--epochs', type=int, default=100, help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=True, help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step', help='lr scheduler')
    parser.add_argument('--step', type=int, default=40, help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, default=None, help='random seed')
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096, help='num of points to use')
    parser.add_argument('--seg_path', type=str, default='data/stanford_indoor3d/', help='Segment Data Path')
    parser.add_argument('--workers', type=int, default=8, help='number of workers')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test')

    args = parser.parse_args()
    args.exp_name = args.model + "_" + args.exp_name

    # 初始化环境
    _init_(args)

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        if args.cuda:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)

    if args.cuda:
        io.cprint('Using GPU')
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        # Test 逻辑类似，可复用 train 中的加载逻辑或单独写 test(args, io)
        pass