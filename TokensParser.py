import torch

import Tokenizer
from utils import createTargetTokens

class TokensParser:
    def __init__(self, trainPercent):
        torch.manual_seed(342299535)

        inputText = open('input.txt', 'r').read()
        tokenizedText = Tokenizer.encode(inputText)

        trainIndex = int(len(tokenizedText) * trainPercent)

        self._tokenizedTextTrain = tokenizedText[:trainIndex]
        self._tokenizedTextTest = tokenizedText[trainIndex:]

    def getTokenizedDataBatch(self, device, train=True, contextCount=8, batchSize=4):
        xs = []
        for _ in range(batchSize):
            if train:
                startIndex = torch.randint(0, len(self._tokenizedTextTrain) - contextCount, (1,))[0].item()
                xs.append(self._tokenizedTextTrain[startIndex: startIndex + contextCount + 1])
            else:
                startIndex = torch.randint(0, len(self._tokenizedTextTest) - contextCount, (1,))[0].item()
                xs.append(self._tokenizedTextTest[startIndex: startIndex + contextCount + 1])
        return createTargetTokens(xs, device=device)