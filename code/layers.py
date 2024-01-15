#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
# Distributed under terms of the MIT license.

"""
This script contains layers used in AllSet and all other tested methods.
"""

import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import Parameter

from dgl.nn.pytorch.conv import GATConv, GATv2Conv, GINConv, GraphConv, APPNPConv, SGConv


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', device='cpu', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm
        #self.device = device
        #self.lamda = lamda
        #self.learn_lamda = llamda
        
        #if in_channels != hidden_channels:
        #    self.transform = nn.Linear(in_channels, hidden_channels)
        #else:
        #    self.transform = None
        
        '''
        if(self.learn_lamda):
            raw_lamda = torch.Tensor([self.lamda])
            raw_lamda = raw_lamda.to(self.device)
            self.raw_lamda = torch.nn.Parameter(data=raw_lamda, requires_grad=True)
        '''
        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        '''
        if(self.learn_lamda):
            raw_lamda = torch.Tensor([self.lamda])
            raw_lamda = raw_lamda.to(self.device)
            self.raw_lamda = torch.nn.Parameter(data=raw_lamda, requires_grad=True)
        '''
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ == 'Identity'):
                normalization.reset_parameters()
        '''
        if self.transform is not None:
            self.transform.reset_parameters()
        '''

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class Res_GAT_layer(nn.Module):
    def __init__(self, args, nhead, in_features, out_features):
        super(Res_GAT_layer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.args = args
        if(args.conv_type == 'GATConv'):
            self.conv = GATConv(in_features, out_features, num_heads = nhead)
        elif(args.conv_type == 'GATv2Conv'):
            self.conv = GATv2Conv(in_features, out_features, num_heads = nhead)

    def reset_parameters(self):
        self.W.reset_parameters()
        self.conv.reset_parameters()
        
    def forward(self, x, x0, g, alpha, beta):
        x = self.conv(g, x)
        
        x = x.mean(1)
        xi = (1-alpha) * x + alpha * x0
        x = (1-beta) * xi + beta * self.W(xi)

        return x

class GAT_layer2(nn.Module):
    def __init__(self, args, device, nhead, in_features, out_features):
        super(GAT_layer2, self).__init__()
        self.args = args
        self.step = args.step
        if(args.conv_type == 'GATConv'):
            self.conv = GATConv(in_features, out_features, num_heads = nhead)
        elif(args.conv_type == 'GATv2Conv'):
            self.conv = GATv2Conv(in_features, out_features, num_heads = nhead)
        self.encoder = nn.Linear((args.hidden*nhead), args.hidden)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.encoder.reset_parameters()
        
    def forward(self, x, x0, g, alpha):
        for _ in range(self.step):
            x = self.conv(g, x)
            x = x.reshape(x0.shape[0], -1)
            x = self.encoder(x)
            x = (1 - alpha) * x + alpha * x0
        return x


class GAT_layer(nn.Module):
    def __init__(self, args, nhead, in_features, out_features):
        super(GAT_layer, self).__init__()
        self.args = args
        self.step = args.step
        if(args.conv_type == 'GATConv'):
            self.conv = GATConv(in_features, out_features, num_heads = nhead)
        elif(args.conv_type == 'GATv2Conv'):
            self.conv = GATv2Conv(in_features, out_features, num_heads = nhead)

    def reset_parameters(self):
        self.conv.reset_parameters()
        
    def forward(self, x, x0, g, alpha):
        for _ in range(self.step):
            x = self.conv(g, x)
            x = x.mean(1)
            x = (1 - alpha) * x + alpha * x0
        return x

class SparseLinear(nn.Module):
    r"""
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # init.ones_(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        # wb=torch.sparse.mm(input,self.weight.T).to_dense()+self.bias
        wb=torch.sparse.mm(input,self.weight.T)
        if self.bias is not None:
            out = wb + self.bias
        else:
            out = wb
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class GAT_ph_layer(nn.Module):
    def __init__(self, args, nhead, in_features, out_features):
        super(GAT_ph_layer, self).__init__()
        self.args = args
        self.step = args.step
        if(args.conv_type == 'GATConv'):
            self.conv = GATConv(in_features, out_features, num_heads = nhead)
        elif(args.conv_type == 'GATv2Conv'):
            self.conv = GATv2Conv(in_features, out_features, num_heads = nhead)

    def reset_parameters(self):
        self.conv.reset_parameters()
        
    def forward(self, x, x0, g, alpha, lamda):
        for _ in range(self.step):
            x_hat = self.conv(g, x)
            x_hat = x_hat.mean(1)
            x = (1 - alpha) * x + alpha * x_hat + alpha * lamda * x0
        return x
    
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output