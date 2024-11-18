import torch.nn

class FeedForward(torch.nn.Module):
    def __init__(self, vocab_size):
        super(FeedForward, self).__init__()
        self.c_fc = torch.nn.Linear(vocab_size, 3072)
        self.c_proj = torch.nn.Linear(3072, vocab_size)
        self.gelu = torch.nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
