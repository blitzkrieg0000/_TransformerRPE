#!/usr/bin/env python
"""
    Anna Huang(2018) tarafından önerilen Shaw(2018)'ın RPE uygulamasının daha optimize edilmiş hali
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPosMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V linear katmanları
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # relative position embeddings
        self.rel_pos_emb = nn.Parameter(torch.randn(2*seq_len-1, self.head_dim))


    def forward(self, x):
        B, L, D = x.size()
        
        # 1. Q,K,V hesapla
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # B,H,L,D
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 2. Normal attention skorları (QKᵀ)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [ B, H, L, L ]
        
        # 3. Relative position skorları
        # Q: B,H,L,D  → (B*H,L,D)
        Q_ = Q.contiguous().view(B*self.num_heads, L, self.head_dim)
        
        # (B*H, L, D) x (D, 2L-1) = (B*H, L, 2L-1)
        rel_scores = torch.matmul(Q_, self.rel_pos_emb.transpose(0, 1))
        
        # 4. Skew uygula
        rel_scores = self._Skew(rel_scores)  # (B*H, L, L)
        rel_scores = rel_scores.view(B, self.num_heads, L, L)
        
        # 5. Toplam attention skoru
        attn_scores = (attn_scores + rel_scores) * self.scale
        
        # 6. Softmax ve V ile çarp
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, V)  # B,H,L,D
        
        # 7. Headleri birleştir
        context = context.transpose(1,2).contiguous().view(B, L, D)
        return self.out_proj(context)


    def _Skew(self, x):
        """
        x: (B*H, L, 2L-1)
        return: (B*H, L, L)
        """
        BHL, L, _ = x.shape

        # (B*H, L, 2L-1) → (L, 2L)
        x = F.pad(x, (1, 0))
        flatt = x.flatten(1, 2)
        x = torch.concat([flatt, torch.zeros(BHL, L-1)], dim=1)
        
        # reshape: (B*H, L+1, L)
        x = x.view(BHL, -1, 2*L-1)
        
        x = x[:, :L, (L - 1):]
        return x


if "__main__" == __name__:
    x = torch.randn(2, 10, 512)
    model = RelPosMultiHeadAttention(512, 8, 10)
    y = model(x)
    print("Shape: ", y.shape)
