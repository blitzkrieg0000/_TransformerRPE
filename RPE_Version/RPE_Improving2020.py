# Improve Transformer Models with Better Relative Position (2020)

import torch
import torch.nn as nn
import torch.nn.functional as F

class RelPosAttentionMethod3(nn.Module):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # Relative position embedding (vector-based)
        self.rel_emb = nn.Embedding(2 * max_len - 1, d_model)
        
        self.scale = self.head_dim ** -0.5
        self.max_len = max_len

    def forward(self, x):
        B, L, _ = x.size()
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Relative positions
        pos_ids = torch.arange(L, device=x.device)
        rel_pos = pos_ids[None, :] - pos_ids[:, None] + self.max_len - 1
        rel_emb = self.rel_emb(rel_pos).view(L, L, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

        # Scores: content + relative
        content_score = torch.matmul(q, k.transpose(-2, -1))
        rel_score = torch.einsum("bhld,hldm->bhlm", q, rel_emb)

        attn_score = (content_score + rel_score) * self.scale
        attn_weights = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


class RelPosAttentionMethod4(RelPosAttentionMethod3):
    def __init__(self, d_model, n_heads, max_len):
        super().__init__(d_model, n_heads, max_len)
        # Extra learnable vectors for global biases
        self.u = nn.Parameter(torch.zeros(n_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(n_heads, self.head_dim))

    def forward(self, x):
        B, L, _ = x.size()
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        pos_ids = torch.arange(L, device=x.device)
        rel_pos = pos_ids[None, :] - pos_ids[:, None] + self.max_len - 1
        rel_emb = self.rel_emb(rel_pos).view(L, L, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

        # Term (a): q·k
        content_score = torch.matmul(q + self.u.unsqueeze(1), k.transpose(-2, -1))
        # Term (b): q·r
        rel_score = torch.einsum("bhld,hldm->bhlm", q + self.v.unsqueeze(1), rel_emb)

        attn_score = (content_score + rel_score) * self.scale
        attn_weights = F.softmax(attn_score, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)
