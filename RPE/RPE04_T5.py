# T5-Raffle(2020): (https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)
"""
    RPE dışında model T5 için:
    => LayerNorm bias yok.
    => LayerNorm, residual toplamanın önüne konmuş (Pre-LN).
"""

import math

import torch
import torch.nn as nn
from einops import rearrange
from torch import arange
import torch.nn.functional as F


# https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py#L87
class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=32, max_distance=128, heads=8, scale=1.0, bidirectional=False,):
        super().__init__()
        self.Scale = scale
        self._Bidirectional = bidirectional
        self.NumBuckets = num_buckets
        self.MaxDistance = max_distance
        self.RelativeAttentionBias = nn.Embedding(num_buckets, heads)

        nn.init.normal_(self.RelativeAttentionBias.weight, mean=0.0, std=0.02) # T5


    @staticmethod
    def _RelativePositionBucket(relative_position, bidirectional = True, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret


    @property
    def device(self):
        return next(self.parameters()).device


    def forward(self, q, m):
        device = self.device
        q_pos = arange(m - q, m, dtype = torch.long, device = device)
        k_pos = arange(m, dtype = torch.long, device = device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._RelativePositionBucket(rel_pos, bidirectional=self._Bidirectional, num_buckets = self.NumBuckets, max_distance = self.MaxDistance)
        # print(rp_bucket)
        values = self.RelativeAttentionBias(rp_bucket)
        bias = rearrange(values, "q m h -> h q m")
        return bias * self.Scale


class MHAWithT5Bias(nn.Module):
    def __init__(self, d_model, n_heads, num_buckets=32, max_distance=128, dropout=0.0, bidirectional=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.RelativePositionBias = RelativePositionBias(
            num_buckets=num_buckets, 
            max_distance=max_distance, 
            heads=n_heads, 
            scale=1.0,
            bidirectional=bidirectional
        )

    def forward(self, x, mask=None):
        # x: (B, L, d_model)
        B, L, _ = x.size()
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)  # (B, H, L, Dh)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1,2)

        # content scores
        attn_scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale  # (B,H,L,L)

        # add T5 relative position bias
        bias = self.RelativePositionBias(L, L)
        attn_scores = attn_scores + bias.unsqueeze(0)  # broadcast to (B,H,L,L)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:, None, :, :], float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B,H,L,Dh)
        out = out.transpose(1,2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)



if "__main__" == __name__:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    B, L, D = 2, 16, 128
    n_heads = 8

    x = torch.randn(B, L, D).to(device)
    model = MHAWithT5Bias(d_model=D, n_heads=n_heads, num_buckets=32, max_distance=128).to(device)
    out = model(x)

    print(out.shape)