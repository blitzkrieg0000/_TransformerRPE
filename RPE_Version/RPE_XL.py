# Transformers XL-RPE

import torch
import torch.nn as nn
import math


class TransformerXLRPE(nn.Module):
    def __init__(self, d_model, max_relative_position=512):
        super().__init__()
        self.d_model = d_model
        
        # Learnable weights
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_R = nn.Linear(d_model, d_model, bias=False)
        
        # Learnable bias vectors u, v
        self.u = nn.Parameter(torch.randn(d_model))
        self.v = nn.Parameter(torch.randn(d_model))
        
        self.max_relative_position = max_relative_position
        
        # Precompute sinusoidal relative positional encodings
        self.relative_positions = torch.arange(-max_relative_position, max_relative_position + 1)
        self.sinusoidal_table = self._generate_sinusoidal_embeddings(2 * max_relative_position + 1, d_model)
        
    
    def _generate_sinusoidal_embeddings(self, length, d_model):
        """Transformer sinüs kodlaması (positional encoding)"""
        position = torch.arange(length).unsqueeze(1)  # (length, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        
        embeddings = torch.zeros(length, d_model)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings  # (length, d_model)
    

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Compute query and key projections
        Q = self.W_Q(x)  # (B, L, d)
        K = self.W_K(x)  # (B, L, d)
        
        # Content-based attention term: (Q + u) * K^T
        # u: (d,), expand to (B, L, d)
        Q_u = Q + self.u  # broadcast add
        
        # Attention scores content
        content_scores = torch.matmul(Q_u, K.transpose(-2, -1))  # (B, L, L)
        
        # Position-based attention term
        # Prepare relative position indices matrix (i-j)
        # Shape: (L, L), relative positions clipped to [-max_relative_position, max_relative_position]
        positions = torch.arange(seq_len).unsqueeze(1) - torch.arange(seq_len).unsqueeze(0)
        positions = positions.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position
        
        # Extract relative positional embeddings: (L, L, d)
        # Using precomputed sinusoidal_table
        sinusoidal_rel_embeddings = self.sinusoidal_table[positions].to(x.device)
        
        # Project relative positional embeddings: (L, L, d)
        R = self.W_R(sinusoidal_rel_embeddings)  # linear on last dim
        
        # Position bias vector v: reshape for broadcast (1, L, d)
        v = self.v.unsqueeze(0).unsqueeze(0)  # (1, 1, d)
        
        # Compute (Q + v): (B, L, d)
        Q_v = Q + v  # broadcast add
        
        # Compute position-based scores: batch matmul Q_v (B,L,d) with R (L,L,d) along d dimension
        # Need to do: for each batch and position i, sum over d: (Q_v[i] * R[i,j]) for all j
        # Equivalent to: einsum('bld,ljd->blj')
        
        pos_scores = torch.einsum("bld,ljd->blj", Q_v, R)
        
        # Combine and normalize
        scores = (content_scores + pos_scores) / math.sqrt(d_model)
        
        return scores  # (B, L, L)




