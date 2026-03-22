'''
This script is adapted from the toolkit of below CVPR paper. 
If you find the functions in this script are helpful to you (for the challenge and beyond), please kindly cite the original paper: 

Riqiang Gao, Bin Lou, Zhoubing Xu, Dorin Comaniciu, and Ali Kamen. 
"Flexible-cm gan: Towards precise 3d dose prediction in radiotherapy." 
In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023.

Disclaimer: This is for research purpose only. This is not part of the any existing Siemens Healthineers product.

'''
import torch
import random
import numpy as np
import SimpleITK as sitk

from monai.transforms import (
    Compose,
    CenterSpatialCropd,
    RandFlipd, 
    RandAffined,
    SpatialPadd, 
    RandSpatialCropd
)

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


def tr_refine_augmentation(linear_keys, nearest_keys, out_size):
    KEYS = linear_keys + nearest_keys
    mode = ['linear'] * len(linear_keys) + ['nearest'] * len(nearest_keys)
    return Compose([
        CenterSpatialCropd(keys=KEYS, roi_size=[int(out_size[0] * 1.2), int(out_size[1] * 1.2), int(out_size[2] * 1.2)], allow_missing_keys=False),
        SpatialPadd(keys=KEYS, spatial_size=[int(out_size[0] * 1.2), int(out_size[1] * 1.2), int(out_size[2] * 1.2)], mode="constant", constant_values=0, allow_missing_keys=False),
        RandAffined(keys=KEYS, prob=0.8, rotate_range=(1, 0.2, 0.2), scale_range=(0.1, 0.1, 0.1), mode=mode, padding_mode="zeros", allow_missing_keys=False),
        RandSpatialCropd(keys=KEYS, roi_size=out_size, random_center=True, random_size=False, allow_missing_keys=False), 
        RandFlipd(keys=KEYS, prob=0.4, spatial_axis=0, allow_missing_keys=False), 
        RandFlipd(keys=KEYS, prob=0.4, spatial_axis=1, allow_missing_keys=False), 
        RandFlipd(keys=KEYS, prob=0.4, spatial_axis=2, allow_missing_keys=False), 
    ])

def tt_refine_augmentation(linear_keys, nearest_keys, out_size):
    KEYS = linear_keys + nearest_keys
    return Compose([
        CenterSpatialCropd(keys=KEYS, roi_size=out_size, allow_missing_keys=False),
        SpatialPadd(keys=KEYS, spatial_size=out_size, mode='constant', constant_values=0, allow_missing_keys=False),
    ])

class NormalizedCTAugmentation:
    def __init__(self, 
                 p=0.5, 
                 brightness_mu=0.0,
                 brightness_sigma=0.1,
                 contrast_range=(0.75, 1.25),
                 gamma_range=(0.7, 1.5)):
        self.p = p
        self.brightness_mu = brightness_mu
        self.brightness_sigma = brightness_sigma
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range

    def apply_brightness(self, img):
        if random.random() < self.p:
            noise_value = random.gauss(self.brightness_mu, self.brightness_sigma)
            img = img + noise_value
            return torch.clamp(img, 0.0, 1.0)
        return img

    def apply_contrast(self, img):
        if random.random() < self.p:
            factor = random.uniform(*self.contrast_range)
            mean = img.mean()
            img = (img - mean) * factor + mean
            return torch.clamp(img, 0.0, 1.0)
        return img

    def apply_gamma(self, img):
        if random.random() < self.p:
            gamma = random.uniform(*self.gamma_range)
            img = torch.clamp(img, min=1e-7) 
            img = torch.pow(img, gamma)
            return torch.clamp(img, 0.0, 1.0)
        return img

    def __call__(self, img):
        img = self.apply_brightness(img)
        img = self.apply_contrast(img)
        img = self.apply_gamma(img)
        return img
