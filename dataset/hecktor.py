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

# two stupid cases in HECKTOR dataset: ['CHUP-075', 'MDA-031']
class HECKTORDataset(Dataset):
    def __init__(self, cfig, phase, mixed_precision='no'):
        self.cfig = cfig
        match mixed_precision:
            case 'bf16':
                self.dtype = torch.bfloat16
            case 'fp16':
                self.dtype = torch.float16
            case _:  
               self.dtype = torch.float32
        
        self.hecktor_root = cfig['hecktor_root']
        df = pd.read_csv(cfig['hecktor_csv'])
        df = df.loc[df['CT'] == True]
        df = df.loc[df['PT'] == True]
        df = df.loc[df['GTV'] == True]
        self.cross_validation_idx = cfig['cross_validation_idx']
        if phase == 'train':
            train_folds = [i for i in range(cfig['fold_num']) if i != self.cross_validation_idx]
            df = df.loc[df['cross-validation'].isin(train_folds)]        
        else:
            df = df.loc[df['cross-validation'] == self.cross_validation_idx]
        self.hecktor_list = df['PID'].tolist()
        self.phase = phase

    def __len__(self):
        return len(self.hecktor_list)
    
    def __getitem__(self, index):
        PatientID = self.hecktor_list[index]
        ct_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__CT.nii.gz')
        pt_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__PT.nii.gz')
        gtv_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}.nii.gz')

        ct_ten, params = io.load_image(ct_path, return_tensor=True)
        pt_ten, _      = io.load_image(pt_path, return_tensor=True)
        gtv_ten, _     = io.load_image(gtv_path, return_tensor=True)
        ct_ten = ct_ten.unsqueeze(0)
        pt_ten = pt_ten.unsqueeze(0)
        gtv_ten = gtv_ten.unsqueeze(0)
        gtv_ten[gtv_ten != 1] = 0

        data_dict = {'ct_ten': ct_ten, 'pt_ten': pt_ten, 'gtv_ten': gtv_ten}
        keys_linear = ['ct_ten', 'pt_ten']
        keys_nearest = ['gtv_ten']

        if self.cfig['with_aug'] and self.phase == 'train':
            processing_func = tr_refine_augmentation(keys_linear, keys_nearest, self.cfig['out_size'])
        else:
            processing_func = tt_refine_augmentation(keys_linear, keys_nearest, self.cfig['out_size'])
        
        data_dict = processing_func(data_dict)
        ct_ten = data_dict['ct_ten']
        pt_ten = data_dict['pt_ten']
        gtv_ten = data_dict['gtv_ten']

        ct_ten = ct_ten.clip(self.cfig['down_HU'], self.cfig['up_HU'])
        ct_ten = (ct_ten - ct_ten.mean()) / (ct_ten.std() + 1e-6)
        pt_percentile_low  = torch.quantile(pt_ten, 0.01)
        pt_percentile_high = torch.quantile(pt_ten, 0.99)
        pt_ten = (pt_ten.clip(pt_percentile_low, pt_percentile_high) - pt_percentile_low) / (pt_percentile_high - pt_percentile_low)

        data_dict = dict()  
        data_dict['ct_ten'] = ct_ten
        data_dict['pt_ten'] = pt_ten
        data_dict['gtv_ten'] = gtv_ten
        data_dict['params'] = params
        data_dict['sample_id'] = PatientID
        data_dict['dataset_name'] = 'HECKTOR'
        
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(self.dtype)

        return data_dict
