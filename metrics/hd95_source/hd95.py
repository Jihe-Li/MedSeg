import numpy as np
from .metrics import compute_surface_distances, compute_robust_hausdorff


def compute_hd95(fixeds, movings, moving_warpeds, labels):
    batch_size = moving_warpeds.shape[0]
    total_hd95 = 0
    for i in range(batch_size):
        fixed = fixeds[i].squeeze(0)
        moving = movings[i].squeeze(0)
        moving_warped = moving_warpeds[i].squeeze(0)
        hd95 = []
        for i in labels:
            if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
                hd95.append(np.NAN)
            else:
                hd95.append(compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.))
        mean_hd95 = np.nanmean(hd95)
        total_hd95 += mean_hd95
    return total_hd95 / batch_size

if __name__ == '__main__':

    import SimpleITK as sitk
    import numpy as np


    fix_img = sitk.ReadImage('/root/autodl-tmp/data/OASIS/OASIS_OAS1_0438_MR1/aligned_seg35.nii.gz')
    mov_img = sitk.ReadImage('/root/autodl-tmp/data/OASIS/OASIS_OAS1_0439_MR1/aligned_seg35.nii.gz')

    fix_img = sitk.GetArrayFromImage(fix_img)
    mov_img = sitk.GetArrayFromImage(mov_img)
    
    labels = np.unique(fix_img)[1:]
    mean_hd95, _ = compute_hd95(fix_img, mov_img, mov_img, labels)
    print(mean_hd95)
