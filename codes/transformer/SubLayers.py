import torch
import torch.nn as nn
import torch.nn.init as init
from transformer.Modules import BottleLinear as Linear
from transformer.Modules import ScaledDotProductAttention, LayerNormalization

class MultiHeadAttention(nn.Module):
    def __init__(self,n_head,d_model,d_k,d_v,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Parameter(torch.FloatTensor(n_head,d_model,d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head,d_model,d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head,d_model,d_v))
        
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v,d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)
        
    def forward(self,q,k,v,attn_mask=None):
        d_k,d_v = self.d_k,self.d_v
        n_head = self.n_head
        
        residual = q
        
        mb_size, len_q, d_model = q.size()
        mb_size, len_k, d_model = k.size()
        mb_size, len_v, d_model = v.size()
        
        q_s = q.repeat(n_head,1,1).view(n_head,-1,d_model)
        k_s = k.repeat(n_head,1,1).view(n_head,-1,d_model)
        v_s = v.repeat(n_head,1,1).view(n_head,-1,d_model)
        
        q_s = torch.bmm(q_s,self.w_qs).view(-1,len_q,d_k)
        k_s = torch.bmm(k_s,self.w_ks).view(-1,len_k,d_k)
        v_s = torch.bmm(v_s,self.w_vs).view(-1,len_v,d_v)
        
        outputs,attns = self.attention(q_s,k_s,v_s,attn_mask=attn_mask.repeat(n_head,1,1))
        
        outputs = torch.cat(torch.split(outputs,mb_size,dim=0),dim=-1)
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)
        
        return self.layer_norm(outputs + residual),attns
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_hid,d_inner_hid,dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Conv1d(d_hid,d_inner_hid,1)
        self.w_2 = nn.Conv1d(d_inner_hid,d_hid,1)
        self.layer_norm = LayerNormalization(d_hid)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        residual = x
        outputs = self.relu(self.w_1(x.transpose(1,2)))
        outputs = self.w_2(outputs).transpose(2,1)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs+residual)