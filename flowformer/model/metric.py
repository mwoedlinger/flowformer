import torch


# binary metrics
def cellwise_acc(output, target):
    with torch.no_grad():
        # >0 because no sigmoid is applied after the model
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        correct = torch.sum(pred == target).item()
    return correct / (target.shape[1] * target.shape[0])


def f1_score(output, target, eps=0.00001):
    with torch.no_grad():
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        tp = torch.sum((pred == 1)*(target == 1)).item()
        fp = torch.sum((pred == 1)*(target == 0)).item()
        fn = torch.sum((pred == 0)*(target == 1)).item()

    # if there are no blast cells and none are detected f1_score is one
    if tp + fn == 0 and fp == 0:
        return 1.0
    else:
        return 2*tp/(2*tp + fp + fn + eps)


def precision(output, target, eps=0.00001):
    with torch.no_grad():
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        tp = torch.sum((pred == 1)*(target == 1)).item()
        fp = torch.sum((pred == 1)*(target == 0)).item()
        fn = torch.sum((pred == 0)*(target == 1)).item()

    return tp/(tp + fp + eps)


def recall(output, target, eps=0.00001):
    with torch.no_grad():
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        tp = torch.sum((pred == 1)*(target == 1)).item()
        fp = torch.sum((pred == 1)*(target == 0)).item()
        fn = torch.sum((pred == 0)*(target == 1)).item()

    return tp/(tp + fn + eps)


# To ensure compatibility for older models
cellwise_acc_binary = cellwise_acc
f1_score_binary = f1_score
precision_binary = precision
recall_binary = recall


def mse(output, target):
    with torch.no_grad():
        mse_pred = torch.nn.functional.mse_loss(output, target)

    return mse_pred


def mrd_gt(output, target, eps=0.00001):
    with torch.no_grad():
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        blast = torch.sum((target == 1)).item()
        other = torch.sum((target == 0)).item()

    return blast/(blast + other)


def mrd_pred(output, target, eps=0.00001):
    with torch.no_grad():
        pred = (output > 0).int()
        assert pred.shape == target.shape, f'pred shape {pred.shape} != target shape {target.shape}'

        blast = torch.sum((pred == 1)).item()
        other = torch.sum((pred == 0)).item()

    return blast/(blast + other)
