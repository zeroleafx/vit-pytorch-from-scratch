import torch
from torch import nn

IMG_SIZE = 64
PATCH_SIZE = 8
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE)**2

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, 
                 embedding_dim: int = 256, 
                 patch_size: int = PATCH_SIZE
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=in_channels, 
                                 out_channels=embedding_dim, 
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0
                                 )
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        img_res = x.shape[-1]
        assert img_res % self.patch_size == 0, "Image size must be divisible by patch size"
        return self.flatten(self.patcher(x)).permute(0,2,1)