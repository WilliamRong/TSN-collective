#encoding:utf-8
#测试脚本
import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule
"""
test_models.py是测试模型的入口。

前面模块导入和命令行参数配置方面和训练代码类似
"""
# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics','cad'])
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()

"""
接下来先是根据数据集来确定类别数。然后通过models.py脚本中的TSN类来导入网络结构。
另外如果想查看得到的网络net的各层信息，可以通过net.state_dict()来查看。
checkpoint = torch.load(args.weights)是导入预训练的模型，在PyTorch中，导入模型都是采用torch.load()接口实现，输入args.weights就是.pth文件，也就是预训练模型。
base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint.state_dict().items())}就是读取预训练模型的层和具体参数并存到base_dict这个字典中。
net.load_state_dict(base_dict)就是通过调用torch.nn.Module类的load_state_dict方法，达到用预训练模型初始化net网络的过程。
需要注意的是load_state_dict方法还有一个输入：strict，如果该参数为True，就表示网络结构的层信息要和预训练模型的层信息严格相等，反之亦然，该参数默认是True。
那么什么时候会用到False呢？就是当你只想用预训练网络初始化你的网络的部分层参数或者说你的预训练网络的层信息和你要被初始化的网络的层信息不完全一致，那样就只会初始化层信息相同的层。

"""
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

net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)
"""
接下来关于args.test_crops的条件语句是用来对数据做不同的crop操作：简单crop操作和重复采样的crop操作。
如果args.test_crops等于1，就先resize到指定尺寸（比如从400resize到256），然后再做center crop操作，
最后得到的是net.input_size的尺寸（比如224），注意这里一张图片做完这些crop操作后输出还是一张图片。
如果args.test_crops等于10，那么就调用该项目下的transforms.py脚本中的GroupOverSample类进行重复采样的crop操作，最终一张图像得到10张crop的结果，后面会详细介绍GroupOverSample这个类。
接下来的数据读取部分和训练时候类似，需要注意的是：
1、num_segments的参数默认是25，比训练时候要多的多。
2、test_mode=True，所以在调用TSNDataSet类的__getitem__方法时和训练时候有些差别。

"""
if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
else:
    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.test_list, num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="img_{:05d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"{}_{:05d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)
"""
接下来是设置GPU模式、设置模型为验证模式、初始化数据等。
"""
if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []

"""
eval_video函数是测试的主体，当准备好了测试数据和模型后就通过这个函数进行预测。
输入video_data是一个tuple：(i, data, label)。
data.view(-1, length, data.size(2), data.size(3))是将原本输入为(1,3*args.test_crops*args.test_segments,224,224)
变换到(args.test_crops*args.test_segments,3,224,224)，相当于batch size为args.test_crops*args.test_segments。
然后用torch.autograd.Variable接口封装成Variable类型数据并作为模型的输入。
net(input_var)得到的结果是Variable，如果要读取Tensor内容，需读取data变量，cpu()表示存储到cpu，numpy()表示Tensor转为numpy array，copy()表示拷贝。
rst.reshape((num_crop, args.test_segments, num_class))表示将输入维数（二维）变化到指定维数（三维），
mean(axis=0)表示对num_crop维度取均值，也就是原来对某帧图像的10张crop或clip图像做预测，最后是取这10张预测结果的均值作为该帧图像的结果。
最后再执行一个reshape操作。最后返回的是3个值，分别表示video的index，预测结果和video的真实标签。

"""
def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape(
        (args.test_segments, 1, num_class)
    ), label[0]

"""
开始循环读取数据，每执行一次循环表示读取一个video的数据。在循环中主要是调用eval_video函数来测试。预测结果和真实标签的结果都保存在output列表中。
"""
proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
                                                                    total_num,
                                                                    float(cnt_time) / (i+1)))
"""
接下来要计算video-level的预测结果，这里从np.mean(x[0], axis=0)可以看出对args.test_segments帧图像的结果采取的也是均值方法来计算video-level的预测结果，
然后通过np.argmax将概率最大的那个类别作为该video的预测类别。video_labels则是真实类别。
cf = confusion_matrix(video_labels, video_pred).astype(float)是调用了混淆矩阵生成结果（numpy array），
举个例子，y_true=[2,0,2,2,0,1]，y_pred=[0,0,2,2,0,2]，那么confusion_matrix(y_true, y_pred)的结果就是array([[2,0,0],[0,0,1],[1,0,2]])，每行表示真实类别，每列表示预测类别。
因此cls_cnt = cf.sum(axis=1)表示每个真实类别有多少个video，cls_hit = np.diag(cf)就是将cf的对角线数据取出，表示每个类别的video中各预测对了多少个，
因此cls_acc = cls_hit / cls_cnt就是每个类别的video预测准确率。
np.mean(cls_acc)就是各类别的平均准确率。
最后的if args.save_scores is not None:语句只是用来将预测结果保存成文件。

"""
video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output]

video_labels = [x[1] for x in output]


cf = confusion_matrix(video_labels, video_pred).astype(float)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.mean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
    name_list = [x.strip().split()[0] for x in open(args.test_list)]

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


