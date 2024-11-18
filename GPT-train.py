import time

from transformers import GPT2LMHeadModel, pipeline

from GPT import GPT
import torch

from Tokenizer import encode, decode
from TokensParser import TokensParser
from utils import generateTokens, getLearningRate, copy_from_model

tokenCount = 50257
token_length = 1024
device = 'mps:0'

model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt = GPT(tokenCount, token_length)

gpt.to(device=device)
model.to(device=device)

copy_from_model(gpt, model)
# generator = pipeline('text-generation', model="gpt2", device='mps')
# print(generator("This is the typical error: ", max_length=50, truncation=True))

tokensParser = TokensParser(.9)
allParamsWithGrad = {n: p for n, p in gpt.named_parameters() if p.requires_grad}

optim = torch.optim.AdamW([
    {'params': [p for n, p in allParamsWithGrad.items() if len(p.size()) >= 2], 'weight_decay': .1},
    {'params': [p for n, p in allParamsWithGrad.items() if len(p.size()) < 2], 'weight_decay': 0}
], betas=(0.9, 0.95), eps=1e-8)
gpt = torch.compile(gpt, backend="aot_eager")

tokenBatch = 50000
tokensPerIteration = 1024 * 8
iterationsForBatch = tokenBatch // tokensPerIteration

for i in range(200):
    lr = getLearningRate(i)
    for param in optim.param_groups:
        param['lr'] = lr
    t0 = time.time()
    totalLoss = 0
    for j in range(iterationsForBatch):
        trainingDataBatch = tokensParser.getTokenizedDataBatch(device, train=True, contextCount=token_length,
                                                               batchSize=8)
        xs = trainingDataBatch['xs']
        yPred = gpt(xs, device)
        loss = torch.nn.functional.cross_entropy(yPred.view(-1, tokenCount), trainingDataBatch['ys'].view(-1),
                                                 ignore_index=-1)
        loss /= iterationsForBatch
        loss.backward()
        totalLoss += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
    optim.step()
    torch.mps.synchronize()
    optim.zero_grad(set_to_none=True)
    timeElapsed = time.time() - t0
    print(
        f"step {i}, time elapsed: {timeElapsed:.4f}, Tokens per second: {(tokensPerIteration * iterationsForBatch) / timeElapsed:.2f}, loss: {totalLoss}, norm: {norm}")

testDataBatch = tokensParser.getTokenizedDataBatch(device, train=False, contextCount=1024)
xs = testDataBatch['xs']
yPred = gpt(xs, device)
loss = torch.nn.functional.cross_entropy(yPred.view(-1, tokenCount), testDataBatch['ys'].view(-1))
print(loss.item())

result = generateTokens(device, gpt, 50, encode("This is the typical error: ").to(device=device))
print(decode(result))
