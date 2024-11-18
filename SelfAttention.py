import torch

from AttentionHeadHandler import AttentionHeadHandler


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_size):
        super(SelfAttention, self).__init__()

        self.c_attn = torch.nn.Linear(embedding_size, embedding_size * 3)
        self.c_proj = torch.nn.Linear(embedding_size, embedding_size)
        self.attentionHeadHandler = AttentionHeadHandler(embedding_size, self.c_attn)

    def forward(self, x):
        x = self.attentionHeadHandler.forward(x)
        return self.c_proj(x)
