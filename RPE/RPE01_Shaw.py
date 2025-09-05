import torch
import torch.nn as nn
import torch.nn.functional as F
SEED = torch.Generator().manual_seed(0)


class ShawRelativeAttention(nn.Module):
	def __init__(self, max_relative_position, num_heads, d_k):
		super(ShawRelativeAttention, self).__init__()
		self.max_relative_position = max_relative_position
		self.num_heads = num_heads
		self.d_k = d_k

		# Relative position embedding: [2K + 1, H, d_k]
		self.rel_pos_emb_k = nn.Parameter(torch.randn(2 * max_relative_position + 1, num_heads, d_k, generator=SEED))
		self.rel_pos_emb_v = nn.Parameter(torch.randn(2 * max_relative_position + 1, num_heads, d_k, generator=SEED))


	def forward(self, L):
		# --- Relative Position Attention Scores ---
		# Create relative position indices: [L, L] with range [-K, K]
		positions = torch.arange(L, dtype=torch.int32)
		rel_indices = positions.view(-1, 1) - positions.view(1, -1)  # [L, L]
		rel_indices = rel_indices.clamp(-self.max_relative_position, self.max_relative_position) + self.max_relative_position  # shift to [0, 2K]
		rel_indices = rel_indices.to(dtype=torch.int32)

		# Lookup: [L, L, H, d_k]
		rel_emb_k, rel_emb_v = self.rel_pos_emb_k[rel_indices], self.rel_pos_emb_v[rel_indices] # [L, L, H, d_k]
		return rel_emb_k, rel_emb_v 



class MultiheadAttentionWithRelativePosition(nn.Module):
	def __init__(self, d_model, num_heads, max_relative_position):
		super().__init__()
		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = d_model // num_heads
		self.max_relative_position = max_relative_position

		self.q_proj = nn.Linear(d_model, d_model)
		self.k_proj = nn.Linear(d_model, d_model)
		self.v_proj = nn.Linear(d_model, d_model)
		self.out_proj = nn.Linear(d_model, d_model)

		self.RelativePositionLayer = ShawRelativeAttention(self.max_relative_position, self.num_heads, self.d_k)


	def forward(self, x):
		# x: [B, L, D]
		B, L, _ = x.size()

		# Linear projections and split into heads
		Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
		K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
		V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, L, d_k]

		# --- (1) Dot-product Attention Scores ---
		# [B, H, L, d_k] x [B, H, d_k, L] -> [B, H, L, L]
		content_scores = torch.einsum("bhld, bhmd -> bhlm", Q, K)  # content-based attention

		# --- (2) Relative Position Attention Scores ---
		# Create relative position indices: [L, L] with range [-K, K]
		rel_emb_k, rel_emb_v = self.RelativePositionLayer(L)
		rel_emb_k = rel_emb_k.permute(2, 0, 1, 3)  # [L, L, H, d_k] -> [H, L, L, d_k]


		#! Shaw's Method (Loop + Broadcast) => (Yöntem daha sonra Anna Huang's Method (2018) ile değiştirilmiştir.)
		# Q: [B, H, L, d_k], rel_emb: [H, L, L, d_k]
		# = [ 32, 8, 128, 64 ] x [8, 128, 128, 64]
		# = [ 32, 8, 128, 1, 64 ] x [1, 8, 128, 128, 64]
		# = [ 32, 8, 128, *128, 64 ] * [*32, 8, 128, 128, 64]
		# = [ 32, 8, 128, 128, 64].sum(dim=-1)
		# =>[ 32, 8, 128, 128]
		rel_scores = torch.einsum("bhlm, hldm -> bhld", Q, rel_emb_k)
		
		attn_scores = content_scores + rel_scores  # [B, H, L, L]
		attn_weights = F.softmax(attn_scores / (self.d_k ** 0.5), dim=-1)  # scaled softmax
		svrel = attn_weights @ V
		
		rel_emb_v = rel_emb_v.permute(2, 0, 1, 3)
		attn_output = torch.einsum("bhlm,hlmd->bhld", attn_weights, rel_emb_v)

		attn_output + svrel

		# 5 - Merge Heads
		attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, D]

		return self.out_proj(attn_output)



if "__main__" == __name__:
	DEVICE = torch.device("cuda:0")
	
	batchSize = 4
	seqLen = 128
	embedSize = 512
	numHeads = 8
	maxRelativePosition = 32

	x = torch.randn(batchSize, seqLen, embedSize, generator=SEED) # <- [32, 128, 512]
	attn = MultiheadAttentionWithRelativePosition(d_model=embedSize, num_heads=numHeads, max_relative_position=maxRelativePosition)
	attn = attn.to(DEVICE)
	out = attn(x.to(DEVICE)) # -> [32, 128, 512]
	
	print(out.shape)





