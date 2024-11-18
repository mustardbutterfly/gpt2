import torch
import math
import matplotlib.pyplot as plt


def createTargetTokens(inputTokens, device):
    return {'xs': torch.stack(
        [inputTokensList[:-1] for inputTokensList in inputTokens]
    ).to(device=device), 'ys': torch.stack(
        [inputTokensList[1:] for inputTokensList in inputTokens]
    ).to(device=device)}


def generateTokens(device, model, length, promptTokens):
    tokens = promptTokens
    for _ in range(length):
        with torch.no_grad():
            tokensFormatted = tokens[-1024:].view(1, -1)
            predictions = model(tokensFormatted, device)  # 1 X X+_
            lastCharPredictions = predictions[-1][-1]
            normalizedPredictions = torch.nn.functional.softmax(lastCharPredictions, dim=-1)
            topKPredictions, indices = torch.topk(normalizedPredictions, 50)
            chosenIndex = torch.multinomial(topKPredictions, 1)
            pickedTensor = torch.gather(indices, -1, chosenIndex)
            # pickedTensor = torch.tensor([indices[.item()]],
            #                             device="mps")
            tokens = torch.cat((tokens, pickedTensor), dim=-1)
    return tokens


def getLearningRate(i):
    peakI = 10
    lowI = 100
    peakLeaningRate = 6e-4

    bottomRate = .1 * peakLeaningRate

    if i < peakI:
        ratio = i / (peakI - 1)
        return ratio * peakLeaningRate
    elif i >= lowI:
        return bottomRate
    else:
        totalSteps = lowI - peakI
        ratio = (i - peakI) / totalSteps
        print(ratio)
        assert 0 <= ratio <= 1
        coeff = .5 * (1 + math.cos(math.pi * ratio))
        return bottomRate + coeff * (peakLeaningRate - bottomRate)


# x = [i for i in range(200)]
# y = [getLearningRate(i) for i in x]
# plt.plot(x,y)
# plt.show()

def copy_from_model(gpt, model):
    with torch.no_grad():
        gpt.transformer.wte.weight.copy_(model.transformer.wte.weight)
        gpt.transformer.wpe.weight.copy_(model.transformer.wpe.weight)
        gpt.transformer.ln_f.weight.copy_(model.transformer.ln_f.weight)
        gpt.transformer.ln_f.bias.copy_(model.transformer.ln_f.bias)
        gpt.lm_head.weight.copy_(model.transformer.wte.weight)
        for index in range(12):
            gpt.transformer.h[index].mlp.c_proj.weight.copy_(
                model.transformer.h[index].mlp.c_proj.weight.transpose(-1, -2))
            gpt.transformer.h[index].mlp.c_proj.bias.copy_(model.transformer.h[index].mlp.c_proj.bias)
            gpt.transformer.h[index].mlp.c_fc.weight.copy_(model.transformer.h[index].mlp.c_fc.weight.transpose(-1, -2))
            gpt.transformer.h[index].mlp.c_fc.bias.copy_(model.transformer.h[index].mlp.c_fc.bias)
            gpt.transformer.h[index].ln_1.weight.copy_(model.transformer.h[index].ln_1.weight)
            gpt.transformer.h[index].ln_1.bias.copy_(model.transformer.h[index].ln_1.bias)
            gpt.transformer.h[index].ln_2.weight.copy_(model.transformer.h[index].ln_2.weight)
            gpt.transformer.h[index].ln_2.bias.copy_(model.transformer.h[index].ln_2.bias)
            gpt.transformer.h[index].attn.c_attn.weight.copy_(
                model.transformer.h[index].attn.c_attn.weight.transpose(-1, -2))
            gpt.transformer.h[index].attn.c_proj.weight.copy_(
                model.transformer.h[index].attn.c_proj.weight.transpose(-1, -2))
            gpt.transformer.h[index].attn.c_attn.bias.copy_(
                model.transformer.h[index].attn.c_attn.bias)
            gpt.transformer.h[index].attn.c_proj.bias.copy_(
                model.transformer.h[index].attn.c_proj.bias)
