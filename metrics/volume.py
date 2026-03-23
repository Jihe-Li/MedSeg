import torch

def comp_volume(pred_seg, tar_seg):
    return pred_seg.sum().detach().cpu().item(), tar_seg.sum().detach().cpu().item()

    