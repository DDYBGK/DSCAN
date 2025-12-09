import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # gold is the groudtruth label in the dataloader

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)  # the number of feature_dim of the ouput, which is output channels

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


# create a file and write the text into it:
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y

'''
    pred [B, 2048, 13]
    target[B, N]
    num_classes: 13
'''
def compute_overall_iou(pred, target, num_sem):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []   #类别iou
        for part in range(num_sem):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            '''
                I是统计第shape_idx个样本中，第part类别的点，被正确分类的个数。
                比如batch中的第二个场景，椅子的类别的点被正确分类的个数
            '''
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            '''
                U是统计第shape_idx个样本中，被预测为第part类别的点，或者真实标签为第part类别的点的个数。
                比如batch中的第二个场景，椅子的类别的点被正确分类的个数
            '''
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            #F是统计第shape_idx个样本中，真实标签为第part类别的点的个数。
            F = np.sum(target_np[shape_idx] == part)

            if F != 0:  #如果这个样本中不含有 part 类别的点，则不计算 iou
                iou = I / float(U)    #  该样本中椅子类别的iou
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # 该列表存放了该批次中中所有样本的类平均iou
    return shape_ious   # [batch_size]
