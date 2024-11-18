import torch.nn


class AttentionHeadHandler:

    def __init__(self, embedding_size, attn):  # 768 x 2304(768*3)
        self.nHead = 12  # 768/12 = 64
        self.embedding_size = embedding_size
        self.attn = attn

    def forward(self, x):  # X x 1024 x 768
        B, T, C = x.shape
        attx = self.attn(x)

        query, keys, value = attx.split(768, dim=2)  # X x 1024 x 768

        k = keys.view(B, T, 12, 64).transpose(2, 1)  # X x 12 x 1024 x 64
        v = value.view(B, T, 12, 64).transpose(2, 1)
        q = query.view(B, T, 12, 64).transpose(2, 1)

        dotProductOfAttention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        # trimmedQueryAndKey = q @ k.transpose(-1, -2) * k.size(-1) ** -0.5
        # tensorForQueryAndKeyScaledDown = trimmedQueryAndKey.masked_fill(mask[:, :, T, :T] == 0, float('-inf'))
        # tensorForQueryAndKeyNormalized = torch.softmax(tensorForQueryAndKeyScaledDown, dim=-1)
        # dotProductOfAttention = tensorForQueryAndKeyNormalized @ v

        return dotProductOfAttention.transpose(1, 2).contiguous().view(B, T, C)
