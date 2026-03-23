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
        
        self.hecktor_root = cfig['hecktor_root']
        self.segrap_root = cfig['segrap_root']
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
        self.segrap_list = [i for i in range(120)]
        self.phase = phase

    def __len__(self):
        if self.phase == 'train':
            return len(self.hecktor_list)
        else:
            return len(self.hecktor_list) + len(self.segrap_list)
    
    def __getitem__(self, index):
        if self.phase == 'train':
            PatientID = self.hecktor_list[index]
            ct_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__CT.nii.gz')
            gtv_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}.nii.gz')
            ct_ten, params = io.load_image(ct_path, return_tensor=True)
            gtv_ten, _     = io.load_image(gtv_path, return_tensor=True)
            ct_ten = ct_ten.unsqueeze(0)
            gtv_ten = gtv_ten.unsqueeze(0)
            gtv_ten[gtv_ten != 1] = 0
            dataset_name = 'HECKTOR'

        else:
            if index < len(self.hecktor_list):
                PatientID = self.hecktor_list[index]
                ct_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}__CT.nii.gz')
                gtv_path = os.path.join(self.hecktor_root, PatientID, f'{PatientID}.nii.gz')
                ct_ten, params = io.load_image(ct_path, return_tensor=True)
                gtv_ten, _     = io.load_image(gtv_path, return_tensor=True)
                ct_ten = ct_ten.unsqueeze(0)
                gtv_ten = gtv_ten.unsqueeze(0)
                gtv_ten[gtv_ten != 1] = 0
                dataset_name = 'HECKTOR'
            else:
                PatientID = 'segrap_%04d' % self.segrap_list[index - len(self.hecktor_list)]
                ct_path = os.path.join(self.segrap_root, PatientID, 'image.nii.gz')
                gtv_path = os.path.join(self.segrap_root, PatientID, 'GTVp.nii.gz')
                ct_ten, params = io.load_image(ct_path, return_tensor=True)
                gtv_ten, _     = io.load_image(gtv_path, return_tensor=True)
                ct_ten = ct_ten.unsqueeze(0)
                gtv_ten = gtv_ten.unsqueeze(0).bool().float()
                dataset_name = 'SegRap'

        data_dict = {'ct_ten': ct_ten, 'gtv_ten': gtv_ten}
        keys_linear = ['ct_ten']
        keys_nearest = ['gtv_ten']
        if self.cfig['with_aug'] and self.phase == 'train':
            processing_func = tr_refine_augmentation(keys_linear, keys_nearest, self.cfig['out_size'])
        else:
            processing_func = tt_refine_augmentation(keys_linear, keys_nearest, self.cfig['out_size'])
        
        data_dict = processing_func(data_dict)
        ct_ten = data_dict['ct_ten']
        gtv_ten = data_dict['gtv_ten']
        ct_ten = ct_ten.clip(self.cfig['down_HU'], self.cfig['up_HU'])
        ct_ten = (ct_ten - ct_ten.mean()) / (ct_ten.std() + 1e-6)

        data_dict = dict()  
        data_dict['ct_ten'] = ct_ten
        data_dict['gtv_ten'] = gtv_ten
        data_dict['params'] = params
        data_dict['sample_id'] = PatientID
        data_dict['dataset_name'] = dataset_name

        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(self.dtype)

        return data_dict
