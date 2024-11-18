import tiktoken
import torch

tiktoker = tiktoken.get_encoding('gpt2')


def encode(input):
    return torch.Tensor(tiktoker.encode(input)).long()


def decode(tensors):
    return tiktoker.decode(tensors.tolist())
