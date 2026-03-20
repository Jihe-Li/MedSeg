import torch
import torch.nn.functional as F


def bce_loss_with_logits(logits, target):
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss

def _focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="none"):
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probab = torch.sigmoid(logits)
    p_t = probab * targets + (1 - probab) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def focal_loss_with_logits(logits, target):
    loss = _focal_loss_with_logits(logits, target, alpha=0.25, gamma=2, reduction='mean')
    return loss

def _dice_loss(y_pred, y_true):
    '''Final loss is divided by batch_size and channel.'''
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2. * intersection) / (union + 1e-5)
    dsc = (1 - torch.mean(dsc))
    return dsc

def dice_loss(probab, target):
    loss = _dice_loss(probab, target)
    return loss
