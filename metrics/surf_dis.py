import numpy as np
from .hd95_source.metrics import compute_surface_distances, compute_robust_hausdorff


def comp_surf_dis(pred_seg, gt_seg, percent=100):
    batch_size = gt_seg.shape[0]
    channel_size = gt_seg.shape[1]
    hd_metric, assd_metric = {}, {}
    for channel_id in range(channel_size):
        hd_metric[f"HD_{channel_id:02d}"] = []
        assd_metric[f"ASSD_{channel_id:02d}"] = []

    gt_seg = gt_seg.detach().bool().cpu().numpy()
    pred_seg = pred_seg.detach().bool().cpu().numpy()
    for batch_id in range(batch_size):
        pred_seg_percase, gt_seg_percase = pred_seg[batch_id], gt_seg[batch_id]
        for channel_id in range(channel_size):
            if np.sum(gt_seg_percase[channel_id]) == 0:
                hd_metric[f"HD_{channel_id:02d}"].append(np.nan)
                assd_metric[f"ASSD_{channel_id:02d}"].append(np.nan)
            elif np.sum(pred_seg_percase[channel_id]) == 0:
                hd_metric[f"HD_{channel_id:02d}"].append(np.inf)
                assd_metric[f"ASSD_{channel_id:02d}"].append(np.inf)
            else:
                surface_dis = compute_surface_distances((gt_seg_percase[channel_id]), (pred_seg_percase[channel_id]), np.ones(3))
                hd = compute_robust_hausdorff(surface_dis, percent)
                hd_metric[f"HD_{channel_id:02d}"].append(hd)
                total_dist = (np.sum(surface_dis["distances_gt_to_pred"] * surface_dis["surfel_areas_gt"]) + 
                              np.sum(surface_dis["distances_pred_to_gt"] * surface_dis["surfel_areas_pred"]))
                total_area = np.sum(surface_dis["surfel_areas_gt"]) + np.sum(surface_dis["surfel_areas_pred"])
                assd_metric[f"ASSD_{channel_id:02d}"].append(total_dist / total_area)

    for key, value in hd_metric.items():
        hd_metric[key] = np.nanmean(value).item()
    for key, value in assd_metric.items():
        assd_metric[key] = np.nanmean(assd_metric[key]).item()

    channels_rm = []
    for key, value in hd_metric.items():
        if np.isnan(value):
            channel_id = int(key.split('_')[1])
            channels_rm.append(channel_id)

    for channel_id in channels_rm:
        hd_metric.pop(f'HD_{channel_id:02d}')
        assd_metric.pop(f'ASSD_{channel_id:02d}')

    avg_hd = np.nanmean(list(hd_metric.values())).item()
    hd_metric['HD_avg'] = avg_hd
    avg_assd = np.nanmean(list(assd_metric.values())).item()
    assd_metric['ASSD_avg'] = avg_assd

    return hd_metric, assd_metric
