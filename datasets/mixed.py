import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from utils import io
from .toolkit import *


class MixedDataset(Dataset):
    def __init__(self, cfig, phase, mixed_precision='no'):
        self.cfig = cfig
        match mixed_precision:
            case 'bf16':
                self.dtype = torch.bfloat16
            case 'fp16':
                self.dtype = torch.float16
            case _:  
               self.dtype = torch.float32

        if phase == 'train' or phase == 'valid':
            self.hecktor_root = cfig['hecktor_root']
            df = pd.read_csv(cfig['hecktor_csv'])
            df = df.loc[df['CT'] == True]
            df = df.loc[df['PT'] == True]
            df = df.loc[df['ct2pet'] == phase]
            self.hecktor_list = df['PID'].tolist()
        
        else:
            df = pd.read_csv(cfig['gdphmm_csv'])
            df = df.loc[df['site'] == 1]
            df = df.loc[df['dev_split'] == phase]
            df = df.loc[df['isVMAT'] == True]
            self.gdphmm_list = df['npz_path_local'].tolist()

        self.phase = phase

    def __len__(self):
        if self.phase == 'train' or self.phase == 'valid':
            return len(self.hecktor_list)
        elif self.phase == 'test':
            return len(self.gdphmm_list)
    
    def __getitem__(self, index):
        if self.phase == 'train' or self.phase == 'valid':
            keys = ['ct_ten', 'pt_ten']
            PatientID = self.hecktor_list[index]
            ct_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__CT.nii.gz')
            pt_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__PT.nii.gz')

            ct_img = sitk.ReadImage(ct_path)
            pt_img = sitk.ReadImage(pt_path)
            # ct_img = resample_img(ct_img, self.cfig['target_spacing'])
            # pt_img = resample_img(pt_img, self.cfig['target_spacing'])
            params = {
                'origin': ct_img.GetOrigin(),
                'spacing': ct_img.GetSpacing(),
                'direction': ct_img.GetDirection()
            }

            ct_ten = torch.FloatTensor(sitk.GetArrayFromImage(ct_img)).unsqueeze(0)
            pt_ten = torch.FloatTensor(sitk.GetArrayFromImage(pt_img)).unsqueeze(0)
            data_dict = {'ct_ten': ct_ten, 'pt_ten': pt_ten}
            if self.cfig['with_aug'] and self.phase == 'train':
                processing_func = tr_refine_augmentation(keys, self.cfig['out_size'])
            else:
                processing_func = tt_refine_augmentation(keys, self.cfig['out_size'])
            data_dict = processing_func(data_dict)
            ct_ten = data_dict['ct_ten']
            pt_ten = data_dict['pt_ten']

            ct_ten = ct_ten.clip(self.cfig['down_HU'], self.cfig['up_HU'])
            ct_ten = (ct_ten - ct_ten.mean()) / (ct_ten.std() + 1e-6)

            pt_percentile_low  = torch.quantile(pt_ten, 0.01)
            pt_percentile_high = torch.quantile(pt_ten, 0.99)
            pt_ten = (pt_ten.clip(pt_percentile_low, pt_percentile_high) - pt_percentile_low) / (pt_percentile_high - pt_percentile_low)

            data_dict = dict()  
            data_dict['ct_ten'] = ct_ten
            data_dict['pt_ten'] = pt_ten
            data_dict['params'] = params
            data_dict['sample_id'] = PatientID
            data_dict['dataset_name'] = 'HECKTOR'
        
        else:
            data_path = self.gdphmm_list[index]
            ID = data_path.split('/')[-1].replace('.npz', '')
            PatientID = ID.split('+')[0]
            data_npz = np.load(data_path, allow_pickle=True)
            In_dict = dict(data_npz)['arr_0'].item()

            data_dict = {'ct_ten': torch.FloatTensor(In_dict['img']).unsqueeze(0)}
            processing_func = tt_refine_augmentation(['ct_ten'], self.cfig['out_size'])
            data_dict = processing_func(data_dict)

            ct_ten = data_dict['ct_ten']
            ct_ten = ct_ten.clip(self.cfig['down_HU'], self.cfig['up_HU'])
            ct_ten = (ct_ten - ct_ten.mean()) / (ct_ten.std() + 1e-6)
            params = {
                'origin': In_dict['origin'],
                'spacing': In_dict['spacing'],
                'direction': In_dict['direction']
            }

            data_dict = dict()  
            data_dict['ct_ten'] = ct_ten
            data_dict['params'] = params
            data_dict['sample_id'] = PatientID
            data_dict['dataset_name'] = 'GDPHMM'

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(self.dtype)

        return data_dict
