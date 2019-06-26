# -*- coding:utf-8 -*-
import torch  
from torch.autograd import Variable  
import torch.nn as nn  
from graphviz import Digraph
from models import TSN


def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}
 
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
 
    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'
 
    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

 
if __name__ == '__main__':  
    net = TSN(8,3,'RGB')
    x = Variable(torch.randn(64,9, 224, 224))
    y = net(x)  
    g = make_dot(y)  
    g.view()  
  
    params = list(net.parameters())  
    k = 0  
    for i in params:  
        l = 1  
        print("该层的结构：" + str(list(i.size())))  
        for j in i.size():  
            l *= j  
        print("该层参数和：" + str(l))  
        k = k + l  
    print("总参数数量和：" + str(k))




