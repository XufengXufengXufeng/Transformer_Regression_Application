import numpy as np
import torch
import transformer.Constants as Constants

def position_encoding_init(n_position,d_pos_vec):
    position_enc = np.array([
        [pos / np.power(10000,2*(j//2)/d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)
    ])
    position_enc[1:,0::2] = np.sin(position_enc[1:,0::2])
    position_enc[1:,1::2] = np.cos(position_enc[1:,1::2])
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

def get_attn_padding_mask(seq_q,seq_k):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    mb_size,len_q = seq_q.size()
    mb_size,len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(Constants.PAD).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(mb_size,len_q,len_k)
    return pad_attn_mask

def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = (seq.size(0),seq.size(1),seq.size(1))
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask)
    return subsequent_mask