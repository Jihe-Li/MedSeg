import torch
import torch.nn.functional as F


def comp_dice(pred_seg, tar_seg):
    B, C, _, _, _ = tar_seg.shape
    inter_areas = (pred_seg * tar_seg).sum((2, 3, 4))
    union_areas = pred_seg.sum((2, 3, 4)) + tar_seg.sum((2, 3, 4))

    dices = 2 * inter_areas / (union_areas + 1e-6)
    return torch.mean(dices).detach().cpu().item()


if __name__ == "__main__":
    pred_seg = torch.randn(1, 1, 10, 10, 10)
    tar_seg = torch.randn(1, 1, 10, 10, 10)
    dice_metric = comp_dice(pred_seg, tar_seg)
    print(dice_metric)
