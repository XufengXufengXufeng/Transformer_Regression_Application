import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import transformer.Constants as Constants
from transformer.Layers import EncoderLayer,DecoderLayer
from transformer.Functions import position_encoding_init,get_attn_padding_mask,get_attn_subsequent_mask
from transformer.Modules import LockedDropout

'''
n_max_seq = TRAIN_PERIODS = 140
d_pos is the dimensions on to which the positions have been projected to
d_pos,d_model,d_inner_hid,n_heads,d_k,and d_v are all hyperparameters
d_model,d_inner_hid,n_heads,d_k,and d_v are all for encoderlayer within which ---
n_head,d_model,d_k,andd_v are for MultiHeadAttention
d_model,d_inner_hid are for PositionwiseFeedForward
ultimately, d_model is for ScaledDotProductAttention and LayerNormalization
'''

class Encoder(nn.Module):
    def __init__(self,n_max_seq,d_model,n_layers=6,n_head=8,d_k=64,d_v=64,
                 d_pos=512,d_inner_hid=1024,dropout=0.1):
        super(Encoder,self).__init__()
        n_position = n_max_seq + 1
        self.position_enc = nn.Embedding(
            n_position,d_pos,padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(
            n_position,d_pos)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model,d_inner_hid,n_head,d_k,d_v,dropout=dropout)
            for _ in range(n_layers)
        ])
    def forward(self,src_seq,src_pos,return_attns=False):
        enc_input = src_seq
        enc_input += self.position_enc(src_pos)
        if return_attns:
            enc_slf_attns = []
        enc_output = enc_input
        enc_slf_attn_mask = get_attn_padding_mask(
            src_seq[:,:,4],src_seq[:,:,4])
        for enc_layer in self.layer_stack:
            enc_output, enc_slt_attn = enc_layer(
                enc_output,slf_attn_mask=enc_slf_attn_mask)
            if return_attns:
                enc_slf_attns += [enc_slf_attn]
        if return_attns:
            return enc_output,enc_slf_attns
        return enc_output

class Decoder(nn.Module):
    def __init__(self,n_max_seq,d_model,n_layers=6,n_head=8,d_k=64,d_v=64,
                d_pos=512,d_inner_hid=1024,dropout=0.1,sub_mask=False):
        super(Decoder,self).__init__()
        self.sub_mask = sub_mask
        n_position = n_max_seq + 1
        self.position_enc = nn.Embedding(
            n_position,d_pos,padding_idx=Constants.PAD)
        self.position_enc.weight.data = position_encoding_init(
            n_position,d_pos)
        self.dropout = nn.Dropout(dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model,d_inner_hid,n_head,d_k,d_v,dropout=dropout)
            for _ in range(n_layers)
        ])
    def forward(self,tgt_seq,tgt_pos,src_seq,enc_output,return_attns=False):
        dec_input = tgt_seq
        '''
        position_enc = self.position_enc(tgt_pos)
        print(position_enc.size())
        #dec_input += position_enc
        '''
        dec_input += self.position_enc(tgt_pos)
        dec_slf_attn_mask = get_attn_padding_mask(
            tgt_seq[:,:,4],tgt_seq[:,:,4])
        if self.sub_mask:
            dec_slf_attn_sub_mask = get_attn_subsequent_mask(tgt_seq[:,:,0])
            dec_slf_attn_mask = torch.gt(dec_slf_attn_mask+dec_slf_attn_sub_mask,0)
        dec_enc_attn_pad_mask = get_attn_padding_mask(
            tgt_seq[:,:,4],src_seq[:,:,4])
        if return_attns:
            dec_slf_attns,dec_enc_attns = [],[]
        dec_output = dec_input
        for dec_layer in self.layer_stack:
            dec_output,dec_slf_attn,dec_enc_attn = dec_layer(
                dec_output,enc_output,
                slf_attn_mask=dec_slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_pad_mask)
            if return_attns:
                dec_slf_attns += [dec_slf_attn]
                dec_enc_attns += [dec_enc_attn]
        if return_attns:
            return dec_output,dec_slf_attns,dec_enc_attns
        return dec_output