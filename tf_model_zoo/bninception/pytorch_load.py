#!/usr/bin/python
#encoding:utf-8
import torch
from torch import nn
from .layer_factory import get_basic_layer, parse_expr
import torch.utils.model_zoo as model_zoo
import yaml


class BNInception(nn.Module):
    def __init__(self, model_path='tf_model_zoo/bninception/bn_inception.yaml', num_classes=101,
                       weight_url='https://yjxiong.blob.core.windows.net/models/bn_inception-9f5701afb96c8044.pth'):
        super(BNInception, self).__init__()

        manifest = yaml.load(open(model_path))
        """
        BNInception类，定义在tf_model_zoo文件夹下的bninception文件夹下的pytorch_load.py中。
        前面当运行self.base_model = getattr(tf_model_zoo, base_model)()，且base_model是‘BNInception’的时候就会调用这个BNInception类的初始化函数__init__。
        manifest = yaml.load(open(model_path))是读进配置好的网络结构（.yml格式），返回的manifest是长度为3的字典，和.yml文件内容对应。
        其中manifest[‘layers’]是关于网络层的详细定义，其中的每个值表示一个层，每个层也是一个字典，包含数据流关系、名称和结构参数等信息。

        """
        layers = manifest['layers']

        self._channel_dict = dict()

        self._op_list = list()
        for l in layers:
            out_var, op, in_var = parse_expr(l['expr'])
            if op != 'Concat':
                id, out_name, module, out_channel, in_name = get_basic_layer(l,
                                                                3 if len(self._channel_dict) == 0 else self._channel_dict[in_var[0]],
                                                                             conv_bias=True)
                """
                然后get_basic_layer函数是用来根据这些参数得到具体的网络层并保存相关信息。
                setattr(self, id, module)是将得到的层写入self的指定属性中，就是搭建层的过程。这样循环完所有层的配置信息后，就搭建好了整个网络。 
                """
                self._channel_dict[out_name] = out_channel
                setattr(self, id, module)
                self._op_list.append((id, op, out_name, in_name))
            else:
                self._op_list.append((id, op, out_var[0], in_var))
                channel = sum([self._channel_dict[x] for x in in_var])
                self._channel_dict[out_var[0]] = channel

        self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))
        """
        建好了网络结构后，另外比较重要的是：self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))这一行，可以分解一下，
        里面的torch.utils.model_zoo.load_url(weight_url)是通过提供的.pth文件的url地址来下载指定的.pth文件，
        在PyTorch中.pth文件就是模型的参数文件，如果你已经有合适的模型了且不想下载，
        那么可以通过torch.load(‘the/path/of/.pth’)导入，因为torch.utils.model_zoo.load_url方法最后返回的时候也是用torch.load接口封装成字典输出。
        self.load_state_dict()则是将导入的模型参数赋值到self中。
        因此不想下载的话可以用checkpoint=torch.load('the/path/of/.pth')和self.load_state_dict(checkpoint)两行代替self.load_state_dict(torch.utils.model_zoo.load_url(weight_url))。

        """
    def forward(self, input):
        data_dict = dict()
        data_dict[self._op_list[0][-1]] = input

        def get_hook(name):

            def hook(m, grad_in, grad_out):
                print(name, grad_out[0].data.abs().mean())

            return hook
        for op in self._op_list:
            if op[1] != 'Concat' and op[1] != 'InnerProduct':
                data_dict[op[2]] = getattr(self, op[0])(data_dict[op[-1]])
                # getattr(self, op[0]).register_backward_hook(get_hook(op[0]))
            elif op[1] == 'InnerProduct':
                x = data_dict[op[-1]]
                data_dict[op[2]] = getattr(self, op[0])(x.view(x.size(0), -1))
            else:
                try:
                    data_dict[op[2]] = torch.cat(tuple(data_dict[x] for x in op[-1]), 1)
                except:
                    for x in op[-1]:
                        print(x,data_dict[x].size())
                    raise
        return data_dict[self._op_list[-1][2]]


class InceptionV3(BNInception):
    def __init__(self, model_path='model_zoo/bninception/inceptionv3.yaml', num_classes=101,
                 weight_url='https://yjxiong.blob.core.windows.net/models/inceptionv3-cuhk-0e09b300b493bc74c.pth'):
        super(InceptionV3, self).__init__(model_path=model_path, weight_url=weight_url, num_classes=num_classes)
