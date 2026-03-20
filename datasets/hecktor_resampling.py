import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy import ndimage


def resample_img(image, 
                 new_spacing, 
                 interpolator=sitk.sitkBSpline3, 
                 pad_value=0):
    """
    将 SimpleITK 图像重采样到指定的体素间距。

    参数:
    - image (sitk.Image): 输入的 SimpleITK 图像。
    - new_spacing (tuple or list): 目标体素间距，例如 (1.0, 1.0, 1.0)。
    - interpolator (sitk.Interpolator): 插值器。常用的有:
        - sitk.sitkNearestNeighbor: 用于分割/标签图。
        - sitk.sitkLinear: 用于灰度图，默认选项。
        - sitk.sitkBSpline: 用于灰度图，效果更平滑。
    - pad_value (float or int): 重采样后图像外部区域的默认填充值。

    返回:
    - sitk.Image: 重采样后的 SimpleITK 图像。
    """
    # 获取原始图像的信息
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    if all(np.array(original_spacing) == np.array(new_spacing)):
        return image

    # Get minimum intensity value
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    min_voxel_value = stats_filter.GetMinimum()
    pad_value = min(pad_value, min_voxel_value)

    # 根据新的间距计算新的图像尺寸
    # 新尺寸 = 旧尺寸 * (旧间距 / 新间距)
    new_size = [
        int(round(orig_size * orig_spacing / n_spacing))
        for orig_size, orig_spacing, n_spacing in zip(original_size, original_spacing, new_spacing)
    ]
    new_size = tuple(new_size)

    # 创建一个 ResampleImageFilter 实例
    resampler = sitk.ResampleImageFilter()

    # 设置输出图像的属性
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(pad_value)
    resampler.SetInterpolator(interpolator)

    # 执行重采样
    resampled_image = resampler.Execute(image)

    return resampled_image


def get_hn_physical_z_bounds(ct_image: sitk.Image, z_offset_upper: float = 20, z_offset_mm: float = 40.0, 
                             min_head_area_mm2: float = 8000.0, min_lung_area_mm2: float = 12000.0):
    """
    计算头颈部 CT 的 Z 轴物理边界 (毫米)
    - 加入物理面积控制，消除 FOV/重采样带来的影响
    """
    img_array = sitk.GetArrayFromImage(ct_image)
    num_z = img_array.shape[0]

    # 获取 XY 平面的单个像素物理面积 (SpacingX * SpacingY)
    spacing = ct_image.GetSpacing()
    pixel_area_mm2 = spacing[0] * spacing[1]

    # 获取每一层的物理 Z 坐标
    z_physical_coords = [ct_image.TransformIndexToPhysicalPoint((0, 0, int(z)))[2] for z in range(num_z)]
    z_indices_top_down = np.argsort(z_physical_coords)[::-1]

    z_upper_bound_idx = None
    lung_apex_z_idx = None

    for z in z_indices_top_down:
        slice_data = img_array[z]
        body_mask = slice_data > -500
        
        # 计算当前层身体掩码的真实物理面积
        body_area_mm2 = body_mask.sum() * pixel_area_mm2

        if z_upper_bound_idx is None and body_area_mm2 > min_head_area_mm2:
            z_upper_bound_idx = z

        if z_upper_bound_idx is not None and body_area_mm2 > min_head_area_mm2:
            solid_body = ndimage.binary_fill_holes(body_mask)
            internal_air = solid_body & (slice_data < -400)
            
            # 计算内部空气的真实物理面积
            internal_air_area_mm2 = internal_air.sum() * pixel_area_mm2
            
            if internal_air_area_mm2 > min_lung_area_mm2:
                lung_apex_z_idx = z
                break  

    # 获取物理上界
    physical_z_upper = z_physical_coords[z_upper_bound_idx] if z_upper_bound_idx is not None else z_physical_coords[0]
    physical_z_upper = physical_z_upper + z_offset_upper
    
    # 获取物理下界 (肺尖位置减去 offset)
    if lung_apex_z_idx is not None:
        apex_physical_z = z_physical_coords[lung_apex_z_idx]
        physical_z_lower = apex_physical_z - z_offset_mm
    else:
        # 如果没找到肺部，默认取最底下
        physical_z_lower = z_physical_coords[z_indices_top_down[-1]]

    # 保证返回值是大值在前还是小值在前并不重要，后续裁剪函数会处理
    return physical_z_upper, physical_z_lower


def crop_multimodal_images(ct_img: sitk.Image, pet_img: sitk.Image, seg_img: sitk.Image, 
                           physical_z_1: float, physical_z_2: float):
    """
    根据给定的 Z 轴物理坐标，同时裁剪 CT, PET 和 Segmentation 图像。
    自动处理不同模态间 Size 和 Spacing 不一致的问题。
    """
    
    def crop_single_image(img: sitk.Image, z_phys_a: float, z_phys_b: float):
        # 1. 构造两个带有目标 Z 物理坐标的 3D 点
        # X 和 Y 的物理坐标可以直接借用图像的 Origin，因为我们只关心 Z 轴的索引
        origin = img.GetOrigin()
        pt_a = (origin[0], origin[1], z_phys_a)
        pt_b = (origin[0], origin[1], z_phys_b)

        # 2. 将物理坐标转换为该图像特有的连续索引 (Continuous Index)
        idx_a = img.TransformPhysicalPointToContinuousIndex(pt_a)[2]
        idx_b = img.TransformPhysicalPointToContinuousIndex(pt_b)[2]

        # 3. 确定安全的裁剪索引范围 (防止计算出的边界超出该模态的实际视野)
        max_z_idx = img.GetSize()[2] - 1
        
        # 四舍五入取整并限制在 [0, max_z_idx] 范围内
        z1 = max(0, min(max_z_idx, int(round(idx_a))))
        z2 = max(0, min(max_z_idx, int(round(idx_b))))

        min_z = min(z1, z2)
        max_z = max(z1, z2)

        # 4. 使用 SimpleITK Image 切片 (注意：SimpleITK Image 的切片顺序是 [X, Y, Z])
        # 这步非常关键，它会自动为你更新新图像的 Origin，保留正确的 Spacing 和 Direction
        cropped_img = img[:, :, min_z : max_z + 1] 
        return cropped_img

    # 分别对三个模态进行裁剪
    cropped_ct = crop_single_image(ct_img, physical_z_1, physical_z_2)
    cropped_seg = crop_single_image(seg_img, physical_z_1, physical_z_2)
    cropped_pet = crop_single_image(pet_img, physical_z_1, physical_z_2)

    return cropped_ct, cropped_pet, cropped_seg


if __name__ == '__main__':

    target_spacing = [2.0, 2.0, 2.0]
    read_root = '/storage/research/artorg_mia/Head_and_Neck/HECKTOR2025_resampled_iso'
    save_root = '/storage/research/artorg_mia/Head_and_Neck/HECKTOR2025_resampled_iso_cropped'
    df = pd.read_csv('data/HECKTOR/meta_data_hecktor.csv')
    df = df.loc[df['CT'] == True]
    df = df.loc[df['PT'] == True]
    df = df.loc[df['GTV'] == True]
    hecktor_list = df['PID'].tolist()

    for PatientID in tqdm(hecktor_list, desc='Resampling', ncols=50):
        ct_path = os.path.join(read_root, PatientID, f'{PatientID}__CT.nii.gz')
        pt_path = os.path.join(read_root, PatientID, f'{PatientID}__PT.nii.gz')
        gtv_path = os.path.join(read_root, PatientID, f'{PatientID}.nii.gz')
        ct_img = sitk.ReadImage(ct_path)
        pt_img = sitk.ReadImage(pt_path)
        gtv_img = sitk.ReadImage(gtv_path)

        # ct_img = resample_img(ct_img, target_spacing)
        # pt_img = resample_img(pt_img, target_spacing)
        # gtv_img = resample_img(gtv_img, target_spacing, interpolator=sitk.sitkNearestNeighbor)

        z_upper, z_lower = get_hn_physical_z_bounds(ct_img, z_offset_upper=20.0, z_offset_mm=20.0)
        z_lower = max(z_lower, z_upper - 500)
        z_lower = min(z_lower, z_upper - 300)
        cropped_ct, cropped_pet, cropped_seg = crop_multimodal_images(ct_img, pt_img, gtv_img, z_upper, z_lower)

        if sitk.GetArrayFromImage(cropped_seg).sum() == 0 or sitk.GetArrayFromImage(cropped_ct).shape[0] < 100:
            tqdm.write(f'Patient {PatientID} has no GTV')

        ct_save_path = os.path.join(save_root, PatientID, f'{PatientID}__CT.nii.gz')
        pt_save_path = os.path.join(save_root, PatientID, f'{PatientID}__PT.nii.gz')
        gtv_save_path = os.path.join(save_root, PatientID, f'{PatientID}.nii.gz')
        os.makedirs(os.path.dirname(ct_save_path), exist_ok=True)
        sitk.WriteImage(cropped_ct, ct_save_path)
        sitk.WriteImage(cropped_pet, pt_save_path)
        sitk.WriteImage(cropped_seg, gtv_save_path)
