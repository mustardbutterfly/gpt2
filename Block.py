import torch

from AttentionHeadHandler import AttentionHeadHandler
from FeedForward import FeedForward
from SelfAttention import SelfAttention

class Block(torch.nn.Module):
    def __init__(self, EmbeddingSize):
        super(Block, self).__init__()
        self.ln_1 = torch.nn.LayerNorm(EmbeddingSize, dtype=torch.float32)
        self.ln_2 = torch.nn.LayerNorm(EmbeddingSize, dtype=torch.float32)
        self.attn = SelfAttention(EmbeddingSize)
        self.mlp = FeedForward(EmbeddingSize)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))
