import os
import sys
import SimpleITK as sitk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.hecktor_resampling import resample_img


def get_topdown_physical_z_bounds(image: sitk.Image, max_slices: int = 192):
    """
    从“头顶方向”(物理 z 最大方向)向下取最多 max_slices 层，
    返回对应的物理 z 上下边界。
    """
    num_z = image.GetSize()[2]
    if num_z <= max_slices:
        return None

    z_physical_coords = [image.TransformIndexToPhysicalPoint((0, 0, z))[2] for z in range(num_z)]
    z_indices_top_down = sorted(range(num_z), key=lambda z: z_physical_coords[z], reverse=True)

    top_idx = z_indices_top_down[0]
    bottom_idx = z_indices_top_down[max_slices - 1]
    return z_physical_coords[top_idx], z_physical_coords[bottom_idx]


def crop_image_by_physical_z(image: sitk.Image, z_phys_a: float, z_phys_b: float):
    """
    按给定物理 z 边界裁剪单个图像。
    使用物理坐标转索引，保证裁剪前后物理空间一致。
    """
    origin = image.GetOrigin()
    idx_a = image.TransformPhysicalPointToContinuousIndex((origin[0], origin[1], z_phys_a))[2]
    idx_b = image.TransformPhysicalPointToContinuousIndex((origin[0], origin[1], z_phys_b))[2]

    max_z_idx = image.GetSize()[2] - 1
    z1 = max(0, min(max_z_idx, int(round(idx_a))))
    z2 = max(0, min(max_z_idx, int(round(idx_b))))
    min_z, max_z = min(z1, z2), max(z1, z2)
    return image[:, :, min_z : max_z + 1]


def crop_ct_gtv_topdown(ct_img: sitk.Image, gtv_img: sitk.Image, max_slices: int = 192):
    bounds = get_topdown_physical_z_bounds(ct_img, max_slices=max_slices)
    if bounds is None:
        return ct_img, gtv_img

    z_upper, z_lower = bounds
    cropped_ct = crop_image_by_physical_z(ct_img, z_upper, z_lower)
    cropped_gtv = crop_image_by_physical_z(gtv_img, z_upper, z_lower)
    return cropped_ct, cropped_gtv


if __name__ == '__main__':
    import os
    import pandas as pd
    import SimpleITK as sitk
    from tqdm import tqdm

    target_spacing = [2.0, 2.0, 2.0]
    read_root = '/storage/research/artorg_mia/Head_and_Neck/SegRap2023/SegRap2023_Training_Set_120cases/SegRap2023_Training_Set_120cases/segrap_%04d'
    save_root = '/storage/research/artorg_mia/Head_and_Neck/SegRap2023/SegRap2023_Training_Set_120cases_resampled_cropped/segrap_%04d'

    for patient_id in tqdm(range(101, 120), desc='Resampling', ncols=50):
        ct_path = os.path.join(read_root % patient_id, 'image.nii.gz')
        gtv_path = os.path.join(read_root % patient_id, 'GTVp.nii.gz')
        ct_img = sitk.ReadImage(ct_path)
        gtv_img = sitk.ReadImage(gtv_path)

        ct_img = resample_img(ct_img, target_spacing)
        gtv_img = resample_img(gtv_img, target_spacing, interpolator=sitk.sitkNearestNeighbor)
        ct_img, gtv_img = crop_ct_gtv_topdown(ct_img, gtv_img, max_slices=192)

        if sitk.GetArrayFromImage(gtv_img).sum() == 0 or sitk.GetArrayFromImage(ct_img).shape[0] < 100:
            tqdm.write(f'Patient {patient_id} has no GTV')
        tqdm.write(f'Patient {patient_id} Image size: {ct_img.GetSize()}, Image spacing: {ct_img.GetSpacing()}')

        ct_save_path = os.path.join(save_root % patient_id, 'image.nii.gz')
        gtv_save_path = os.path.join(save_root % patient_id, 'GTVp.nii.gz')
        os.makedirs(os.path.dirname(ct_save_path), exist_ok=True)
        sitk.WriteImage(ct_img, ct_save_path)
        sitk.WriteImage(gtv_img, gtv_save_path)
