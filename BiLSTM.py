#!/usr/bin/python
#encoding:utf-8

"""
B代表batch size，

L_i代表在batch中第i个序列的长度，L\in R^B是一个长度为B的向量

x(i,0:L_i,0:d_{input})代表在batch中第i个序列，其长度为L_i，每一帧的维度是d_{input}；每一个batch的数据x的矩阵大小为x\in R^{B\times L_{max}\times d_{input}}，其中L_{max}是序列L中的最大值，对于长度不足L_{max}事先应进行补0操作

y(i,0:L_i)代表在batch中第i个序列的类别，每一个batch的数据y的矩阵大小为y\in R^{B\times L_{max}}，其中L_{max}是序列L中的最大值，对于长度不足L_{max}事先应进行补-1操作（避免和0混淆，其实补什么都无所谓，这里只是为了区分）

在这里，我将先使用Pytorch的原生API，搭建一个BiLSTM。先吐槽一下Pytorch对可变长序列处理的复杂程度。处理序列的基本步骤如下：

1.准备torch.Tensor格式的data=x，label=y，length=L，等等
2.数据根据length排序，由函数sort_batch完成
3.pack_padded_sequence操作
4.输入到lstm中进行训练
"""

import torch.nn as nn
import numpy as np

def sort_batch(data,label,length):
    batch_size=data.size(0)
    inx=torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data=data[inx]
    label=label[inx]
    length=label[inx]
    length=list(length.numpy())
    return (data,label,length)

class Net(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,num_layers,biFlag,dropout=0.5):
        # input_dim 输入特征维度d_input
        # hidden_dim 隐藏层的大小
        # output_dim 输出层的大小（分类的类别数）
        # num_layers LSTM隐藏层的层数
        # biFlag 是否使用双向
        super(Net,self).__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.output_dim=output_dim
        self.num_layers=num_layers
        if(biFlag):self.bi_num=2
        else:self.bi_num=1
        self.biFlag=biFlag
        # 根据需要修改device
        self.device=torch.device("cuda")

        #定义LSTM网络的输入，输出，层数，是否batch_first，dropout比例，是否双向
        self.layer1=nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,\
                            num_layers=num_layers,batch_first=True,\
                            dropout=dropout,bidirectional=biFlag)
        # 定义线性分类层，使用logsoftmax输出
        self.layer2=nn.Sequential(
            nn.Linear(hidden_dim*self.bi_num,output_dim),
            nn.LogSoftmax(dim=2)
        )

        self.to(self.device)


    def init_hidden(self,batch_size):
        # 定义初始的hidden state
        return (torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers*self.bi_num,batch_size,self.hidden_dim).to(self.device))

    def forward(self,x,y,length):
        # 输入原始数据x，标签y，以及长度length
        # 准备
        batch_size=x.size(0)
        max_length=torch.max(length)
        # 根据最大长度截断
        x=x[:,0:max_length,:];y=y[:,0:max_length]
        x,y,length=sort_batch(x,y,length)
        x,y=x.to(self.device),y.to(self.device)
        # pack sequence
        x=pack_padded_sequence(x,length,batch_first=True)

        # run the network
        hidden1=self.init_hidden(batch_size)
        out,hidden1=self.layer1(x,hidden1)
        # out,_=self.layerLSTM(x) is also ok if you don't want to refer to hidden state
        # unpack sequence
        out,length=pad_packed_sequence(out,batch_first=True)
        out=self.layer2(out)
        # 返回正确的标签，预测标签，以及长度向量
        return y,out,length



