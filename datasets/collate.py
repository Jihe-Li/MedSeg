import torch


def collate_fn(batch):
    keys = list(batch[0].keys())
    out_batch = {}
    for key in keys:
        out_batch[key] = []
    for case in batch:
        for key, value in case.items():
            out_batch[key].append(value)

    for key, value in out_batch.items():
        if 'ten' in key:
            out_batch[key] = torch.stack(out_batch[key], dim=0)

    return out_batch
