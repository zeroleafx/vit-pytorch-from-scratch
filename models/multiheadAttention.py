import torch
from torch import nn

class MultiheadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 4, attn_dropout: float = 0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.msa = nn.MultiheadAttention(embed_dim=embedding_dim, 
                                         num_heads=num_heads, 
                                         dropout=attn_dropout, 
                                         batch_first=True
                                         )

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.msa(query=x, key=x, value=x, need_weights=False)
        return attn_output