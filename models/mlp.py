import torch
from torch import nn

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int=256, mlp_size: int=1024, mlp_dropout: float=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=mlp_dropout)
        )
    def forward(self, x):
        return self.mlp(self.layer_norm(x))