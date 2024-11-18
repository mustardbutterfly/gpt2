import torch

from Block import Block

embedding_size = 768


class GPT(torch.nn.Module):
    def __init__(self, tokensCount, token_length):
        super(GPT, self).__init__()
        self.token_length = token_length
        wte = torch.nn.Embedding(tokensCount, embedding_size)
        self.transformer = torch.nn.ModuleDict({
            'wte': wte,
            'wpe': torch.nn.Embedding(token_length, embedding_size),
            'h': torch.nn.ModuleList([
                Block(embedding_size) for _ in range(12)
            ]),
            'ln_f': torch.nn.LayerNorm(embedding_size)
        })
        self.lm_head = torch.nn.Linear(embedding_size, tokensCount, bias=False)
        self.lm_head.weight = wte.weight

    def forward(self, x, device):
        embeddedInput = self.transformer.wte(x)
        embeddingPosInput = self.transformer.wpe(torch.arange(x.size(1), device=device))
        input = embeddedInput + embeddingPosInput
        for module in self.transformer.h:
            input = module(input)
        normalized = self.transformer.ln_f(input)
        return self.lm_head(normalized)
