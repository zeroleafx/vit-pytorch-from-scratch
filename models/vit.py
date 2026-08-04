import torch
from torch import nn
from .patchEmbedding import PatchEmbedding
from .transformerEncoder import TransformerEncoderBlock

IMG_SIZE = 64
PATCH_SIZE = 8

class ViT(nn.Module):
    def __init__(self, image_size: int=IMG_SIZE, 
                 in_channels: int=3, 
                 patch_size: int=PATCH_SIZE, 
                 num_transformer_layers: int=8, 
                 embedding_dim: int=256, 
                 mlp_size: int=1024, 
                 num_heads: int=4, 
                 attn_dropout: float=0, 
                 mlp_dropout: float=0.1,
                 embedding_dropout: int=0.1,
                 num_classes: int=200):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"

        self.num_patches = (image_size * image_size) // patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim))

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim
                                              )
        
        # Create the transformer encoder block
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size, mlp_dropout=mlp_dropout, attn_dropout=attn_dropout) for _ in range(num_transformer_layers)])

        # Create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
    
    def forward(self, x):
        BATCH_SIZE = x.shape[0]
        class_token = self.class_embedding.expand(BATCH_SIZE, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])
        return x
