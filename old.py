import torch

from Block import Block
from TokensParser import TokensParser

device = "mps"
EmbeddingSize = 64
numberOfChars = 65
contextCount = 8

tokensContext = TokensParser(.9)

model = Block(EmbeddingSize)
model.to(device)
embeddingLayer = torch.nn.Embedding(numberOfChars, EmbeddingSize)
embeddingLayer.to(device)
embeddingPosLayer = torch.nn.Embedding(contextCount, EmbeddingSize)
embeddingPosLayer.to(device)
outputLayer = torch.nn.Linear(EmbeddingSize, numberOfChars)
outputLayer.to(device)
layerNormOutput = torch.nn.LayerNorm(EmbeddingSize)
layerNormOutput.to(device)

optim = torch.optim.AdamW(model.parameters())

for i in range(200000):
    trainingDataBatch = tokensContext.getTokenizedDataBatch(train=True, contextCount=contextCount)
    xs = trainingDataBatch['xs']
    embeddedInput = embeddingLayer(xs)
    embeddingPosInput = embeddingPosLayer(torch.arange(contextCount, device=device))
    input = embeddedInput + embeddingPosInput  # 4 x 8 x 64
    trainedInput = model(input)
    yPred = outputLayer(layerNormOutput(trainedInput))  # remaining layers from the paper
    loss = torch.nn.functional.cross_entropy(yPred.view(-1, numberOfChars), trainingDataBatch['ys'].view(-1))
    if i % 1000 == 0:
        print(loss.item())
    model.zero_grad()
    loss.backward()
    optim.step()

testDataBatch = tokensContext.getTokenizedDataBatch(train=False, contextCount=contextCount)
xs = testDataBatch['xs']
embeddedInput = embeddingLayer(xs)
embeddingPosInput = embeddingPosLayer(torch.arange(contextCount, device=device))
trainedInput = model(embeddedInput + embeddingPosInput)
yPred = outputLayer(layerNormOutput(trainedInput))  # remaining layers from the paper
loss = torch.nn.functional.cross_entropy(yPred.view(-1, numberOfChars), testDataBatch['ys'].view(-1))
print(loss.item())

# generateText(100, tokensContext)


