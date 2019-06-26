#encoding:utf-8

#网络结构构建脚本
from torch import nn

import numpy as np
from ops.basic_ops import ConsensusModule, Identity
from transforms import *
from torch.nn.init import normal, constant
#from dataset import
import scipy.io
import csv
import pandas as pd
import os

global count, iter, epoch
count=0
iter=0
epoch=0

class TSN(nn.Module):#继承nn.Module
    def __init__(self, num_class, num_segments, modality,
                 base_model='resnet101', new_length=None,
                 consensus_type='avg', before_softmax=True,
                 dropout=0.8,
                 crop_num=1, partial_bn=True):
        super(TSN, self).__init__()
        self.modality = modality#特征种类，从RGB，Flow，RGBDiff中选
        self.num_segments = num_segments#默认为3,视频分割成的段数
        self.reshape = True#是否reshape
        self.before_softmax = before_softmax#模型是否在softmax前的意思？
        self.dropout = dropout#dropout参数
        self.crop_num = crop_num#裁剪数量?
        self.consensus_type = consensus_type#聚集函数G的设置，论文中有5种，这里有avg,max,topk,cnn,rnn
        if not before_softmax and consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            self.new_length = 1 if modality == "RGB" else 5#RGB取1张图，Flow取5张图？
        else:
            self.new_length = new_length

        print(("""
Initializing TSN with base model: {}.
TSN Configurations:
    input_modality:     {}
    num_segments:       {}
    new_length:         {}
    consensus_module:   {}
    dropout_ratio:      {}
        """.format(base_model, self.modality, self.num_segments, self.new_length, consensus_type, self.dropout)))
        #导入模型初始化
        self._prepare_base_model(base_model)
        feature_dim = self._prepare_tsn(num_class)
        #根据特征不同对网络结构进行修改
        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")

        self.consensus = ConsensusModule(consensus_type)#ConsensusModule就是聚集函数模块定义的关键

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = partial_bn
        if partial_bn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            self.new_fc = nn.Linear(feature_dim, num_class)
        #如果设置中没有dropout层，设置last_layer为Linear层，也就是作为全连接层使用
        #如果设置中有dropout层,设置last_layer为dropout层，再接一层Linear作为全连接层
        #下面的语句是设置last_layer的w和b
        std = 0.001
        if self.new_fc is None:
            normal(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal(self.new_fc.weight, 0, std)
            constant(self.new_fc.bias, 0)
        return feature_dim
    """
    _prepare_tsn方法。feature_dim是网络最后一层的输入feature map的channel数。
    接下来如果有dropout层，那么添加一个dropout层后连一个全连接层，否则就直接连一个全连接层。
    setattr是torch.nn.Module类的一个方法，用来为输入的某个属性赋值，一般可以用来修改网络结构，
    以setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))为例，输入包含3个值，分别是基础网络，要赋值的属性名，要赋的值，
    一般而言setattr的用法都是这样。因此当这个setattr语句运行结束后，self.base_model.last_layer_name这一层就是nn.Dropout(p=self.dropout)。 
    最后对全连接层的参数（weight）做一个0均值且指定标准差的初始化操作，参数（bias）初始化为0。getattr同样是torch.nn.Module类的一个方法，
    与为属性赋值方法setattr相比，getattr是获得属性值，一般可以用来获取网络结构相关的信息，以getattr(self.base_model, self.base_model.last_layer_name)为例，输入包含2个值，分别是基础网络和要获取值的属性名。

    """
    def _prepare_base_model(self, base_model):

        if 'resnet' in base_model or 'vgg' in base_model:
            self.base_model = getattr(torchvision.models, base_model)(True)#获取base_model的属性值
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            #调整input_size,mean,std为适合网络的值
            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length

        elif base_model == 'BNInception':
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            # getattr模块的使用：getattr(tf_model_zoo, base_model)()类似tf_model_zoo.BNInception()
            # 因为要根据base_model的不同指定值来导入不同的网络，所以用getattr模块。导入模型之后就是一些常规的配置信息了。
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)

        elif 'inception' in base_model:
            import tf_model_zoo
            self.base_model = getattr(tf_model_zoo, base_model)()
            self.base_model.last_layer_name = 'classif'
            self.input_size = 299
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        #重写train()来冻结BN参数
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):#如果base_model里出现一次BatchNorm2d,加一
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()#python中的eval()是将字符串转换成表达式计算，但不填参数是什么意思？

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
                        #梯度需求置否，使更新停止
    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):#提取各层参数
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())#将参数转换成list保存进ps
                conv_cnt += 1#计数卷积层数量，出现2d或1d卷积就计一次
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])#提取第一个卷积层的weight
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])#如果参数是2，即bias不是0,取bias
                else:
                    normal_weight.append(ps[0])#取各层参数
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):#同上，这是全连接层（线性变换层）
                ps = list(m.parameters())
                normal_weight.append(ps[0])
                if len(ps) == 2:
                    normal_bias.append(ps[1])
                  
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))#extend()用于在list末尾追加另一个序列的多个值，append()是添加单一对象
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))


        #保存参数

        #first_conv_weight_array=np.array(first_conv_weight)
        #print(first_conv_weight)
        #np.savetxt('./first_conv_weight.txt',fmt=['%s']*first_conv_weight_array.shape[1],newline='\n')
        #print('Finishing saving txt file')



        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]


        """
        也许可以在forward中加hook来得到中间变量
        """
        """
    def output_hook(self,input):

        global count
        global glb_baseout

        glb_baseout = torch.cat((glb_baseout, input), 0)

        count += 1

        if (count == 3):
            glb_out = glb_baseout[1:]
            glb_baseout = torch.zeros(1, 1024)
            count = 0
        return glb_out
        """





    def forward(self, input):#TSN类的forward函数定义了模型前向计算过程，也就是TSN的base_model+consensus结构
        sample_len = (3 if self.modality == "RGB" else 2) * self.new_length#RGB为3*new_length,其他为2*new_length，有何用意？

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length#RGBDiff也是3*new_length，作何解释?
            input = self._get_diff(input)#RGB过一遍_get_diff函数，得到RGBDiff的input

        #print(np.shape(input))
        #print(np.shape(input.view((-1, sample_len) + input.size()[-2:])))

        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        print(base_out.shape)

        global count,iter,epoch
        if count==3:
            count=0
            iter+=1
        if iter==16:
            epoch+=1
            iter=0
        #print(epoch)

        print('forward_%d'%(count))
        base_file = open('./output/output_%d_%d_%d.csv' % (epoch,iter,count), 'a+')
        writer = csv.writer(base_file)
        count += 1
        if epoch == 79:
            writer.writerows(base_out.data.cpu().numpy())
            #writer.writerows('\n')
        base_file.close()






        #try output base_out



        if self.dropout > 0:#是否加入dropout
            base_out = self.new_fc(base_out)
            #print(np.shape(base_out))


        if not self.before_softmax:#如果未加softmax，加softmax?
            base_out = self.softmax(base_out)
            #print(base_out)
            #print(np.shape(base_out))

        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
            #tensor.view(-1,3)为将base_out维度重置为(-1,3)的tensor,-1的位置根据实际size改变，实际为原size积/3，比如原来为(6,9),view(-1,3)后变为(18,3)
            #size()[1:]返回第二个维度之后的维度(包括第二个维度自身)，比如(10,3,4,5)，就返回(3,4,5)



        #print(np.array(base_out).shape)
        #print(np.shape(base_out))
        output = self.consensus(base_out)#经过consensus函数，得到最终输出
        #print(output)
        return output.squeeze(1)#squeeze(1)与squeeze()一个作用，去掉tensor中为1的维度，比如(10,1,2,1)变为(10,2)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2#RGB，RGBDiff为3，Flow为2，此处input_c是通道数
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        #具体size较复杂，一般为input.view((-1,3,2,3)+input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()#clone()与copy()一样，复制一个tensor，开辟一块新的内存给这个tensor
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()#复制一份input_view,其中第三个维度取第二之后的维度,比如(10,1,2,3,4,5)[:,:,1:,:,:,:]即为(10,1,1,3,4,5)
            #也就是第三个维度是通道数，keep_rgb即为3，否则减1变为2，即flow的通道数
        for x in reversed(list(range(1, self.new_length + 1))):#倒着取，也就是[2,1]
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]#求RGBDiff,R-B,G-R,B-G作为新的RGB
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
                #Flow差值,一般的Flow是(h色调,s饱和度,v亮度)三通道的，这个网络提取时只提取了(s，v)双通道的Flow?

        return new_data


    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]#lambda为匿名函数，这里表示参数为x，isinstance判断modules中有多少卷积层,取第一个作为第一个卷积层的index
        #filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
        #接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
        conv_layer = modules[first_conv_idx]#定义一个卷积层，值为原来的第一个卷积层
        container = modules[first_conv_idx - 1]#定义原来第一个卷积层前的一层为container

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]#克隆卷积层参数
        kernel_size = params[0].size()#params[0]是weight
        new_kernel_size = kernel_size[:1] + (2 * self.new_length, ) + kernel_size[2:]#卷积核大小的设置，下面注释有说
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()#contiguous()保证内存连续性，使得view()函数可以正常使用
        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)#重新设置卷积层
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7] # remosve .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model
    """
    前面提到如果输入不是RGB，那么就要修改网络结构，这里以models.py脚本中TSN类的_construct_flow_model方法介绍对于optical flow类型的输入需要修改哪些网络结构。
    conv_layer是第一个卷积层的内容，params 包含weight和bias，kernel_size就是(64,3,7,7)，
    因为对于optical flow的输入，self.new_length设置为5，所以new_kernel_size是(63,10,7,7)。new_kernels是修改channel后的卷积核参数，主要是将原来的卷积核参数复制到新的卷积核。
    然后通过nn.Conv2d来重新构建卷积层。new_conv.weight.data = new_kernels是赋值过程。
    """
    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat((params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                                    1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
