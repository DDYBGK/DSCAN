from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from util.S3DISDataLoader import PartNormalDataset, S3DISDataset
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
import sklearn.metrics as metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #C:\Users\13456\Desktop\论文\LCHNet\semantic_seg
ROOT_DIR = BASE_DIR  #C:\Users\13456\Desktop\论文\LCHNet\semantic_seg
sys.path.append(os.path.join(ROOT_DIR, 'model')) #将当前工作目录下的 model 目录加入到 sys.path

def inplace_relu(m):
    classname = m.__class__.__name__ #获取模块m的类名,.__class__返回模块m的类对象， .__name__获取类对象名称
    # 字符串的 find 方法查找子字符串 'ReLU' 的位置，若找到返回起始索引；若没找到则返回-1
    if classname.find('ReLU') != -1:
        m.inplace=True #这意味着该 ReLU 操作将在输入张量上直接进行修改，而不会创建新的张量 是一种优化手段


'''
    设置 NumPy 的随机种子，以确保后续调用 NumPy 的随机数生成函数时，结果是可复现的。
    随机种子都基于当前时间变化，从而避免不同运行之间的结果完全一致。将两者相加作为随机
    种子，既保证了线程间的独立性，又引入了时间的变化性。
'''
def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))

#将类别与数字标签对应起来
classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}  #enumerate返回列表的索引、再返回classes元素值
seg_classes = class2label
seg_label_to_cat = {}  #创建空字典
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

#######废弃不用########
classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']

'''创建检查点文件夹函数'''
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)

'''
    参数：m是pytorch模型的模块，作用：对模型中的不同类型的层进行权重和偏置的初始化
'''
def weight_init(m):
    if isinstance(m, torch.nn.Linear):  #检查当前模块m是否是torch.nn.Linear
        torch.nn.init.xavier_normal_(m.weight)  #使用正态分布初始化方法将Linear层的权重初始化
        if m.bias is not None:          #检查该层是否存在偏置
            torch.nn.init.constant_(m.bias, 0)  #将偏置初始化为常数0
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)  #将二维批量归一化层的缩放参数初始化为1
        torch.nn.init.constant_(m.bias, 0)    #将二维批量归一化层的偏置初始化为0
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)  #将一维批量归一化层的缩放参数初始化为1
        torch.nn.init.constant_(m.bias, 0)    #将一维批量归一化层的偏置初始化为0


def train(args, io):

    # ============= Model ===================
    num_part = 50
    num_semantic = 13
    device = torch.device("cuda" if args.cuda else "cpu")
    #实例化一个神经网络模型并将其移动到指定的设备（GPU或CPU）
    '''
        models.__dict_[]表示取函数或者类的指针,
        这时候后边加上（num_semantic）表示是其参数
        最后.to(device)将这个模型的运行设备进行了指定
        实例化了模型 model就是实例化对象
    '''
    model = models.__dict__[args.model](num_semantic).to(device) #定义模型运行设备CPU/GPU
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)   #将模型包装为支持多GPU并行训练的形式
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    '''
    Resume or not
    这段代码判断是否从之前保存的模型检查点恢复训练
    如果命令行参数 args.resume 为 True，表示需要从之前的训练状态恢复。否则，直接打印 "Training from scratch..."，表示从头开始训练。
    '''
    if args.resume:
        state_dict = torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name,
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Resume training model...")
        print(torch.load("checkpoints/%s/best_insiou_model.pth" % args.exp_name).keys())
    else:
        print("Training from scratch...")

    # =========== Dataloader =================
    '''加载数据集'''
    train_dataset = S3DISDataset(split='train', data_root=args.seg_path, num_point=args.seg_point,
                                 test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("The number of training data is:%d", len(train_dataset))
    test_dataset = S3DISDataset(split='test', data_root=args.seg_path, num_point=args.seg_point,
                                test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("The number of training data is:%d", len(test_dataset))
    trainDataLoader = DataLoader(train_dataset, batch_size=args.seg_batch_size, shuffle=True,
                                                  num_workers=4, drop_last=True, pin_memory=True,
                                                  worker_init_fn=worker_init_fn)
    testDataLoader = DataLoader(test_dataset, batch_size=args.seg_batch_size, shuffle=False,
                                                 num_workers=4, drop_last=True)
    weights = torch.Tensor(train_dataset.labelweights).cuda()  # 类别不平衡（某些类别的数据比其他类别的数据少得多）可能会导致模型性能下降，某些类别在数据集中出现的频率较低，则会赋予这些类别更高的权重，以避免模型偏向于频率较高的类别。
    # log_string("The number of training data is: %d" % len(train_dataset))
    # log_string("The number of test data is: %d" % len(test_dataset))
    ###########部分分割数据加载
    #train_data = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    #print("The number of training data is:%d", len(train_data))

    #test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    #print("The number of test data is:%d", len(test_data))

    #train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,drop_last=True)

    #test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0
    best_class_iou = 0
    best_instance_iou = 0
    num_part = 50
    num_sem= 13
    #num_classes = 16

    for epoch in range(args.epochs):

        train_epoch(trainDataLoader, model, opt, scheduler, epoch, num_sem, io)  #减少了一个参数

        test_metrics, total_per_cat_iou = test_epoch(testDataLoader, model, epoch, num_part, io)#减少了一个参数

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/%s/best_acc_model.pth' % args.exp_name)

        # 2. when get the best instance_iou, save the model:
        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            io.cprint('Max instance iou:%.5f' % best_instance_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou}
            torch.save(state, 'checkpoints/%s/best_insiou_model.pth' % args.exp_name)

        # 3. when get the best class_iou, save the model:
        # first we need to calculate the average per-class iou
        class_iou = 0
        for cat_idx in range(16):
            class_iou += total_per_cat_iou[cat_idx]
        avg_class_iou = class_iou / 16
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            # print the iou of each class:
            for cat_idx in range(16):
                io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
            io.cprint('Max class iou:%.5f' % best_class_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou}
            torch.save(state, 'checkpoints/%s/best_clsiou_model.pth' % args.exp_name)

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max instance iou:%.5f' % best_instance_iou)
    io.cprint('Final Max class iou:%.5f' % best_class_iou)
    # save last model
    state = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou}
    torch.save(state, 'checkpoints/%s/model_ep%d.pth' % (args.exp_name, args.epochs))

#应该把train_epoch删除
def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, io):

    ########原始的精度计算
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

        # Note: We do not need to calculate per_class iou during training

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch+1, train_loss * 1.0 / count,
                                                                       metrics['accuracy'], metrics['shape_avg_iou'])
    io.cprint(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(16):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    io.cprint(outstr)

    return metrics, final_total_per_cat_iou


def test(args, io):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # Try to load models
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    model = models.__dict__[args.model](num_part).to(device)
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/%s/best_%s_model.pth" % (args.exp_name, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 50
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(16):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=10,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()
    args.exp_name = args.model+"_"+args.exp_name

    _init_()

    if not args.eval:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_train.log' % (args.exp_name))
    else:
        io = IOStream('checkpoints/' + args.exp_name + '/%s_test.log' % (args.exp_name))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
