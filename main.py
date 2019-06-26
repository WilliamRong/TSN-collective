#encoding:utf-8
#main.py训练脚本,是训练模型的入口
import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import csv
import copy
import numpy as np

from torch.nn.utils import clip_grad_norm

import models

from dataset import TSNDataSet
from models import TSN #导入TSN
from transforms import *
from opts import parser #导入配置参数
#from visualize import make_dot


best_prec1 = 0


def main():

    torch.set_printoptions(precision=6)

    global args, best_prec1
    args = parser.parse_args()
    #导入参数设置数据集类数量
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    elif args.dataset == 'cad':
        num_class = 8
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    """
    #导入模型，输入包含分类的类别数：
    # num_class；args.num_segments表示把一个video分成多少份，对应论文中的K，默认K=3；
    # 采用哪种输入：args.modality，比如RGB表示常规图像，Flow表示optical flow等；
    # 采用哪种模型：args.arch，比如resnet101，BNInception等；
    # 不同输入snippet的融合方式：args.consensus_type，比如avg等；
    # dropout参数：args.dropout。
    """
    model = TSN(num_class, args.num_segments, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    """
    接着main函数的思路，前面这几行都是在TSN类中定义的变量或者方法，model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()是设置多GPU训练模型。
    args.resume这个参数主要是用来设置是否从断点处继续训练，比如原来训练模型训到一半停止了，希望继续从保存的最新epoch开始训练，
    因此args.resume要么是默认的None，要么就是你保存的模型文件（.pth）的路径。
    其中checkpoint = torch.load(args.resume)是用来导入已训练好的模型。
    model.load_state_dict(checkpoint[‘state_dict’])是完成导入模型的参数初始化model这个网络的过程，load_state_dict是torch.nn.Module类中重要的方法之一。

    """
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    cudnn.benchmark = True

    # Data loading code
    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5
    """
    接下来是main函数中的第二部分：数据导入。首先是自定义的TSNDataSet类用来处理最原始的数据，返回的是torch.utils.data.Dataset类型，
    一般而言在PyTorch中自定义的数据读取类都要继承torch.utils.data.Dataset这个基类，比如此处的TSNDataSet类，然后通过重写初始化函数__init__和__getitem__方法来读取数据。
    torch.utils.data.Dataset类型的数据并不能作为模型的输入，还要通过torch.utils.data.DataLoader类进一步封装，
    这是因为数据读取类TSNDataSet返回两个值，第一个值是Tensor类型的数据，第二个值是int型的标签，
    而torch.utils.data.DataLoader类是将batch size个数据和标签分别封装成一个Tensor，从而组成一个长度为2的list。
    对于torch.utils.data.DataLoader类而言，最重要的输入就是TSNDataSet类的初始化结果，其他如batch size和shuffle参数是常用的。通过这两个类读取和封装数据，后续再转为Variable就能作为模型的输入了。

    """

    train_loader = torch.utils.data.DataLoader(
        TSNDataSet("",  args.train_list,num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   transform=torchvision.transforms.Compose([
                       train_augmentation,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=3, pin_memory=True)


    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+"{}_{:05d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=3, pin_memory=True)
    """
    接下来就是main函数的第三部分：训练模型。这里包括定义损失函数、优化函数、一些超参数设置等，然后训练模型并在指定epoch验证和保存模型。
    adjust_learning_rate(optimizer, epoch, args.lr_steps)是设置学习率变化策略，args.lr_steps是一个列表，里面的值表示到达多少个epoch的时候要改变学习率，
    在adjust_learning_rate函数中，默认是修改学习率的时候修改成当前的0.1倍。
    train(train_loader, model, criterion, optimizer, epoch)就是训练模型，输入包含训练数据、模型、损失函数、优化函数和要训练多少个epoch。
    最后的if语句是当训练epoch到达指定值的时候就进行一次模型验证和模型保存，args.eval_freq这个参数就是用来控制保存的epoch值。
    prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))就是用训练好的模型验证测试数据集。
    最后的save_checkpoint函数就是保存模型参数（model）和其他一些信息，这里我对源代码做了修改，希望有助于理解，该函数中主要就是调用torch.save(mode, save_path)来保存模型。
    模型训练函数train和模型验证函数validate函数是重点，后面详细介绍。

    """
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    '''
    optimizer = torch.optim.SGD(policies,
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    '''
    # try Adam instead.
    optimizer=torch.optim.Adam(policies,args.lr)

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)


        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader))

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best)



"""
train函数是模型训练的入口。首先一些变量的更新采用自定义的AverageMeter类来管理，后面会介绍该类的定义。然后model.train()是设置为训练模式。 
for i, (input, target) in enumerate(train_loader) 是数据迭代读取的循环函数，
具体而言，当执行enumerate(train_loader)的时候，是先调用DataLoader类的__iter__方法，该方法里面再调用DataLoaderIter类的初始化操作__init__。
而当执行for循环操作时，调用DataLoaderIter类的__next__方法，在该方法中通过self.collate_fn接口读取self.dataset数据时就会调用TSNDataSet类的__getitem__方法，从而完成数据的迭代读取。
读取到数据后就将数据从Tensor转换成Variable格式，然后执行模型的前向计算：output = model(input_var)，得到的output就是batch size*class维度的Variable；
损失函数计算： loss = criterion(output, target_var)；准确率计算： prec1, prec5 = accuracy(output.data, target, topk=(1,5))；模型参数更新等等。
其中loss.backward()是损失回传， optimizer.step()是模型参数更新。

"""

def for_hook(module, input, output):
    #print(module)
    for val in input:
        print("input val:",val)
    for out_val in output:
        print("output val:", out_val)

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if args.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)

    # switch to train mode
    model.train()


    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target_tmp=copy.deepcopy(target)#copy target

        for j,idx in enumerate(target):
            target[j] = (idx % 1000) / 100

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        #handle = model.register_forward_hook(for_hook)
        # compute output
        output= model(input_var)

        #print(handle)
        #handle.remove()

        if(epoch==args.epochs-1):
            """
            base_file = open('./output/output_%d_%d.csv' % (epoch,i), 'a+')
            writer = csv.writer(base_file)
            writer.writerows(models.glb_out)
            base_file.close()
            """
            with open('./output/target_%d_%d.txt'%(epoch,i),'a+') as target_file:
                target_file.write(str(target_tmp))
                target_file.close()

        #print(output.size(),output,target_var.size(),target_var)
        loss = criterion(output, target_var)

        #output_list=list(output)
        #output_table.append(output_list)
        #print(output)
        #print(target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if args.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), args.clip_gradient)
            if total_norm > args.clip_gradient:
                print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


    #f=open('./output/output_%s.csv'%(args.modality),'a+')
    #writer=csv.writer(f)
    #writer.writerows(base_output)
    #writer.writerow('\n')
    #f.close()

    #csv_saver = pd.DataFrame(np.array(base_output).reshape(-1, 1024))
    #csv_saver.to_csv("./output_%s.csv" % (args.modality), index=False, header=False)
    #print("output saved!")

    #file = open('output.txt','w');

    #file.write(str(output_table));

    #file.close();

"""
验证函数validate基本上和训练函数train类似，主要有几个不同点。
先是model.eval()将模型设置为evaluate mode，
其次没有optimizer.zero_grad()、loss.backward()、optimizer.step()等损失回传或梯度更新操作。

"""
def validate(val_loader, model, criterion, iter, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

        target_tmp=copy.deepcopy(target)

        for j,idx in enumerate(target):
            target[j]=(idx%1000)/100

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)

        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open('./output/target_val_%d.txt' % (i), 'w') as val_file:
            val_file.write(str(target_tmp))
            val_file.close()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = '_'.join((args.snapshot_pref, args.modality.lower(), filename))
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join((args.snapshot_pref, args.modality.lower(), 'model_best.pth.tar'))
        shutil.copyfile(filename, best_name)

"""
在train函数中采用自定义的AverageMeter类来管理一些变量的更新。
在初始化的时候就调用的重置方法reset。
当调用该类对象的update方法的时候就会进行变量更新，当要读取某个变量的时候，可以通过对象.属性的方式来读取，
比如在train函数中的top1.val读取top1准确率。

"""
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

"""
准确率计算函数。输入output是模型预测的结果，尺寸为batch size*num class；
target是真实标签，长度为batch size。
这二者都是Tensor类型，具体而言前者是Float Tensor，后者是Long Tensor。
batch_size = target.size(0)是读取batch size值。
_, pred = output.topk(maxk, 1, True, True)这里调用了PyTorch中Tensor的topk方法，
第一个输入maxk表示你要计算的是top maxk的结果；
第二个输入1表示dim，即按行计算（dim=1）；
第三个输入True完整的是largest=True，表示返回的是top maxk个最大值；
第四个输入True完整的是sorted=True，表示返回排序的结果，主要是因为后面要基于这个top maxk的结果计算top 1。
target.view(1, -1).expand_as(pred)先将target的尺寸规范到1*batch size，然后将维度扩充为pred相同的维度，
也就是maxk*batch size，比如5*batch size，然后调用eq方法计算两个Tensor矩阵相同元素情况，
得到的correct是同等维度的ByteTensor矩阵，1值表示相等，0值表示不相等。
correct_k = correct[:k].view(-1).float().sum(0)通过k值来决定是计算top k的准确率，
sum(0)表示按照dim 0维度计算和，最后都添加到res列表中并返回。

"""
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""


    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()

