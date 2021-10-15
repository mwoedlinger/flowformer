from torch.nn.functional import nll_loss, cross_entropy, binary_cross_entropy_with_logits
from torch.nn.functional import mse_loss as mse_loss_torch
import torch


def ce_loss(output, target, weight=torch.tensor([0.1, 1.0])):
    """
    output: [B, C, *] TODO: test
    target: [B, *]
    Crossentropy loss. Combines nn.LogSoftmax() and nn.NLLLoss(), i.e. works with raw output (no logits).
    Expects a one-hot encoded vector for output and a number (the class index) for target.
    """
    return cross_entropy(output, target.long(), weight=weight.to(output.device))

def nll_loss(output, target):
    """
    output: [B, C, *] (values in [0,1]) TODO: test
    target: [B, *]
    Negative log likelihood loss. Expects logits, i.e. does not require a softmax at the end of the model.
    Expects a one-hot encoded vector for output and a number (the class index) for target.
    """
    return nll_loss(output, target)

def focal_loss(output, target, alpha=0.1, gamma=4, eps=0.00001):
    """
    output: [B, *] TODO: test
    target: [B, *]
    Focal loss. Puts more weight on cells with y=1.
    Expects a number (0 or 1) for both output and target.
    """
    p = torch.nn.Sigmoid()(output)
    fl_ye1 = -alpha * (1 - p) ** gamma * torch.log(p + eps)
    fl_ye0 = -(1 - alpha) * p ** gamma * torch.log(1 - p + eps)

    return torch.sum(target * fl_ye1 + (1 - target) * fl_ye0)

def bce_loss(output, target):
    """
    output: [B, *]
    target: [B, *]

    Binary cross entropy. Expects a number (0 or 1) for both output and target.
    """
    return binary_cross_entropy_with_logits(output, target)


def mse_loss(output, target):
    """
    output: [B, *]
    target: [B, *]

    Mean squared error loss.
    """
    return mse_loss_torch(output, target)



