# XL Modelindeki Relative Positional Encoding (RPE)

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.scale = 1 / (self.d_head ** 0.5)
        self.max_len = max_len

        # Projections
        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)  # W_{k,E}
        self.v_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)
        self.r_proj = nn.Linear(d_model, n_head * self.d_head, bias=False)  # W_{k,R}

        # Learnable global bias (u, v)
        self.u_bias = nn.Parameter(torch.Tensor(n_head, self.d_head))  # u
        self.v_bias = nn.Parameter(torch.Tensor(n_head, self.d_head))  # v

        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer("rel_pos_enc", self._GetSinusoidalRPE(max_len, d_model))
        self._reset_parameters()


    def _GetSinusoidalRPE(self, length, d_model):
        inv_freq = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pos_seq = torch.arange(-length + 1, length).float()
        sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
        emb = torch.zeros(2 * length - 1, d_model)
        emb[:, 0::2] = torch.sin(sinusoid_inp)
        emb[:, 1::2] = torch.cos(sinusoid_inp)
        return emb  # [2*length-1, d_model]
    
    
    def _reset_parameters(self):
        nn.init.normal_(self.u_bias, 0.0, 0.02)
        nn.init.normal_(self.v_bias, 0.0, 0.02)


    def forward(self, w, attn_mask=None):
        """
        w: input tensor, shape [B, Lq, d_model]
        attn_mask: optional mask
        """
        B, Lq, _ = w.size()

        # Relative positional encoding tensöründen ihtiyacımız olan kısmı alıyoruz
        r = self.rel_pos_enc[self.max_len - Lq : self.max_len + Lq - 1, :]  # [2Lq-1, d_model]

        # Projeksiyonlar
        q = self.q_proj(w).view(B, Lq, self.n_head, self.d_head)     # [B, Lq, H, Dh]
        k = self.k_proj(w).view(B, Lq, self.n_head, self.d_head)     # [B, Lk, H, Dh]
        v = self.v_proj(w).view(B, Lq, self.n_head, self.d_head)     # [B, Lk, H, Dh]
        r = self.r_proj(r).view(2 * Lq - 1, self.n_head, self.d_head)  # [Lr, H, Dh]

        # Transpose heads first
        q = q.transpose(1, 2)  # [B, H, Lq, Dh]
        k = k.transpose(1, 2)  # [B, H, Lk, Dh]
        v = v.transpose(1, 2)  # [B, H, Lk, Dh]
        r = r.transpose(1, 0)  # [H, Lr, Dh]

        # (a) Content-based attention
        AC = torch.matmul(q + self.u_bias.unsqueeze(1), k.transpose(-2, -1))  # [B, H, Lq, Lk]

        # (b) Content-dependent positional bias
        BD = torch.matmul(q + self.v_bias.unsqueeze(1), r.transpose(-2, -1))  # [B, H, Lq, Lr]
        BD = self._rel_shift(BD)  # Relative shift

        attn_score = (AC + BD) * self.scale

        if attn_mask is not None:
            attn_score = attn_score.masked_fill(attn_mask[:, None, :, :], float('-inf'))

        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        attn_vec = torch.matmul(attn_prob, v)  # [B, H, Lq, Dh]

        attn_vec = attn_vec.transpose(1, 2).contiguous().view(B, Lq, -1)

        return attn_vec


    def _rel_shift(self, x):
        B, H, Lq, Lr = x.size()
        zero_pad = torch.zeros((B, H, Lq, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)  # [B, H, Lq, Lr+1]

        x_padded = x_padded.view(B, H, Lr + 1, Lq)
        x = x_padded[:, :, 1:].view(B, H, Lq, Lr)

        x = x[:, :, :, (Lr - Lq):]  # Son slice ile hizalama

        return x



if __name__ == "__main__":
    BATCH = 4
    Lq = 128
    DIM_MODEL = 512
    HEAD_SIZE = 8

    x = torch.randn(BATCH, Lq, DIM_MODEL)

    attn_layer = RelPartialLearnableMultiHeadAttn(HEAD_SIZE, DIM_MODEL, max_len=512)
    output = attn_layer(x)

    print("Output shape:", output.shape)
