import torch

import torch.nn as nn
import torch.nn.functional as F

import math 

from layers import *

class SGC(nn.Module):
    def __init__(self, args):
        super(SGC, self).__init__()
        
        nlayers = args.num_layers
        nhidden = args.hidden
        variant = args.variant
        nfeat = args.num_features
        nclass = args.num_classes
        dropout = args.dropout
        nhead = args.heads
        
        self.convs = nn.ModuleList()
        self.convs.append(SGConv(nhidden, nhidden, nlayers))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for i,con in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = con(adj, x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        
        nlayers = args.num_layers
        nhidden = args.hidden
        variant = args.variant
        nfeat = args.num_features
        nclass = args.num_classes
        dropout = args.dropout
        nhead = args.heads
        
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GATConv(nhidden, nhidden, num_heads = nhead))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for i,con in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = con(adj, x)
            x = x.mean(1)
            x = self.act_fn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)
    
class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        
        nlayers = args.num_layers
        nhidden = args.hidden
        variant = args.variant
        nfeat = args.num_features
        nclass = args.num_classes
        dropout = args.dropout
        nhead = args.heads
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConv(nhidden, nhidden, norm='none'))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        
    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for i,con in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(con(adj, x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)

class GCNII(nn.Module):
    def __init__(self, args):
        super(GCNII, self).__init__()
        
        nlayers = args.num_layers
        nhidden = args.hidden
        variant = args.variant
        nfeat = args.num_features
        nclass = args.num_classes
        dropout = args.dropout
        alpha = args.alpha
        lamda = args.lamda
        
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for fc in self.fcs:
            fc.reset_parameters()
        
    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner, adj, _layers[0], self.lamda, self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)
