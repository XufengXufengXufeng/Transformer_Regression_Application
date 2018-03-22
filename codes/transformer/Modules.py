import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self,d_in,d_out,bias=True):
        super(Linear,self).__init__()
        self.linear = nn.Linear(d_in,d_out,bias=bias)
        init.xavier_normal(self.linear.weight)
    def forward(self,x):
        return self.linear(x)
    
class Bottle(nn.Module):
    def forward(self,inputs):
        if len(inputs.size())<=2:
            return super(Bottle,self).forward(inputs)
        size = inputs.size()[:2]
        out = super(Bottle,self).forward(inputs.view(size[0]*size[1],-1))
        return out.view(size[0],size[1],-1)
    
class BottleLinear(Bottle,Linear):
    pass

class BottleSoftmax(Bottle,nn.Softmax):
    pass

class LayerNormalization(nn.Module):
    def __init__(self,d_hid,eps=1e-3):
        super(LayerNormalization,self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid),requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid),requires_grad=True)
        
    def forward(self,z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z,keepdim=True,dim=-1)
        sigma = torch.std(z,keepdim=True,dim=-1)
        ln_out = (z-mu.expand_as(z))/(sigma.expand_as(z)+self.eps)
        ln_out = ln_out*self.a_2.expand_as(ln_out)+self.b_2.expand_as(ln_out)
        return ln_out
    
class BatchBottle(nn.Module):
    def forward(self,inputs):
        if len(inputs.size()) <= 2:
            return super(BatchBottle,self).forward(inputs)
        size = inputs.size()[1:]
        out = super(BatchBottle,self).forward(inputs.view(-1,size[0]*size[1]))
        return out.view(-1,size[0],size[1])
    
class BottleLayerNormalization(BatchBottle,LayerNormalization):
    pass

class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_model,attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temper = np.power(d_model,0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = BottleSoftmax(dim=1)
        
    def forward(self,q,k,v,attn_mask=None):
        attn = torch.bmm(q,k.transpose(1,2))/self.temper
        
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(),\
            'Attention mask shape {} mismatch'\
            'with Attention logit tensor shape'\
            '{}.'.format(attn_mask.size(),attn.size())
            
            attn.data.masked_fill_(attn_mask,-float('inf'))
            
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn,v)
        
        return output,attn   

class LockedDropout(nn.Module):
    def __init__(self,batch_first=False):
        super().__init__()
        self.batch_first = batch_first
        
    def forward(self,x,dropout=0.5):
        if not self.training or not dropout:
            return x
        if self.batch_first:
            m = x.data.new(x.size(0),1,x.size(2)).bernoulli_(1-dropout)
        else:
            m = x.data.new(1,x.size(1),x.size(2)).bernoulli_(1-dropout)
        mask = Variable(m,requires_grad=False)/(1-dropout)
        mask = mask.expand_as(x)
        return mask*x