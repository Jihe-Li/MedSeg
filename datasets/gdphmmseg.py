from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
import json

from .toolkit import *


HaN_OAR_LIST = ['BrachialPlexus', 'Posterior_Neck', 'Larynx', 'OralCavity', 'Parotids', 'PharynxConst', 'Trachea', 'SpinalCord_05', 'Submandibular']
HaN_OAR_DICT = {HaN_OAR_LIST[i]: (i+1) for i in range(len(HaN_OAR_LIST))}

class GDPHMMDataset(Dataset):
    def __init__(self, cfig, phase, mixed_precision='no'):
        '''
        phase: train, validation, or testing 
        cfig: the configuration dictionary
        
            train_bs: training batch size
            val_bs: validation batch size
            num_workers: the number of workers when call the DataLoader of PyTorch
            
            csv_root: the meta data file, include patient id, plan id, the .npz data path and some conditions of the plan. 
            scale_dose_dict: path of a dictionary. The dictionary includes the prescribed doses of the PTVs. 
            pat_obj_dict: path of a dictionary. The dictionary includes the ROIs (PTVs and OARs) names used in optimization. 
            
            down_HU: bottom clip of the CT HU value. 
            up_HU: upper clip of the CT HU value. 
            denom_norm_HU: the denominator when normalizing the CT. 
            
            in_size & out_size: the size parameters used in data transformation. 

            norm_oar: True or False. Normalize the OAR channel or not. 
            CatStructures: True or False. Concat the PTVs and OARs in multiple channels, or merge them in one channel, respectively. 

            dose_div_factor: the value used to normalize dose. 
            
        '''
        
        self.cfig = cfig
        match mixed_precision:
            case 'bf16':
                self.dtype = torch.bfloat16
            case 'fp16':
                self.dtype = torch.float16
            case _:  
               self.dtype = torch.float32
        df = pd.read_csv(cfig['csv_root'])
        df = df.loc[df['site'] == 1]
        df = df.loc[df['dev_split'] == phase]
        if cfig['VMAT_only'] == True:
            df = df.loc[df['isVMAT'] == True]

        self.phase = phase
        self.seg_list = ['PTVHighOPT'] + HaN_OAR_LIST
        self.data_list = df['npz_path_local'].tolist()
        
        if not self.phase == 'train':
            patient_ids = df['PID'].tolist()
            _, ids = np.unique(patient_ids, return_index=True)
            ids = sorted(ids.tolist())
            self.data_list = [self.data_list[i] for i in ids]

        self.scale_dose_Dict = json.load(open(cfig['scale_dose_dict'], 'r'))
        self.pat_obj_dict = json.load(open(cfig['pat_obj_dict'], 'r'))
        # if self.phase == 'train':
            # self.inten_aug = NormalizedCTAugmentation()

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        ID = self.data_list[index].split('/')[-1].replace('.npz', '')
        PatientID = ID.split('+')[0]
        data_npz = np.load(data_path, allow_pickle=True)
        In_dict = dict(data_npz)['arr_0'].item()

        OAR_LIST = HaN_OAR_LIST
        OAR_DICT = HaN_OAR_DICT
        if self.cfig['with_official_need_list'] == True:
            try:
                need_list = self.pat_obj_dict[ID.split('+')[0]] # list(a.values())[0]
                need_oars = [item for item in need_list if item in OAR_LIST]
            except:
                need_list = OAR_LIST
                need_oars = OAR_LIST
        else:
            need_list = OAR_LIST
            need_oars = [item for item in In_dict.keys() if item in OAR_LIST]

        linear_keys = ['img', 'dose']
        nearest_keys = need_oars + [self.scale_dose_Dict[PatientID]["PTV_High"]['OPTName'], 'Body', 'PTV_Total']
        nearest_keys = list(set(nearest_keys))

        isocenter = In_dict.pop('isocenter')
        In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / self.cfig['denom_norm_HU'] 
        # In_dict['img'] = np.clip(In_dict['img'], self.cfig['down_HU'], self.cfig['up_HU']) / (self.cfig['denom_norm_HU'] * 4) + 0.5
        ori_img_size = list(In_dict['img'].shape)
        params = {'ori_img_size': ori_img_size,
                  'ori_isocenter': [i.astype(np.float32).item() for i in isocenter],
                  'origin': In_dict['origin'],
                  'spacing': In_dict['spacing'],
                  'direction': In_dict['direction']}

        if 'dose' in In_dict.keys():
            ptv_highdose =  self.scale_dose_Dict[PatientID]['PTV_High']['PDose']
            In_dict['dose'] = In_dict['dose'] * In_dict['dose_scale'] 
            PTVHighOPT = self.scale_dose_Dict[PatientID]['PTV_High']['OPTName']
            norm_scale = ptv_highdose / (np.percentile(In_dict['dose'][In_dict[PTVHighOPT].astype('bool')], 3) + 1e-5) # D97
            In_dict['dose'] = In_dict['dose'] * norm_scale / self.cfig['dose_div_factor']
            In_dict['dose'] = np.clip(In_dict['dose'], 0, ptv_highdose * 1.2)
            params['p_dose'] = ptv_highdose
            params['dose_restore_scale'] = (self.cfig['dose_div_factor'] / norm_scale).item()

        if self.phase == 'train':
            if 'with_aug' in self.cfig.keys() and not self.cfig['with_aug']:
                self.spatial_aug = tt_refine_augmentation(linear_keys, nearest_keys, list(self.cfig['in_size']), list(self.cfig['out_size']), isocenter)
            else:
                self.spatial_aug = tr_refine_augmentation(linear_keys, nearest_keys, list(self.cfig['in_size']), list(self.cfig['out_size']), isocenter)
        if self.phase in ['val', 'test', 'valid', 'external_test']:
            self.spatial_aug = tt_refine_augmentation(linear_keys, nearest_keys, list(self.cfig['in_size']), list(self.cfig['out_size']), isocenter)

        rm_keys = []
        for key in In_dict.keys():
            if key in linear_keys + nearest_keys:
                continue
            else:
                rm_keys.append(key)
        for key in rm_keys:
            In_dict.pop(key)
        for key, value in In_dict.items():
            # if key in nearest_keys:
            #     voxel_num = value.sum()
            #     if voxel_num < 8:
            #         In_dict[key] = np.zeros_like(value)
            In_dict[key] = torch.FloatTensor(value).unsqueeze(0)

        In_dict = self.spatial_aug(In_dict)
        _, cat_oar = combine_oar(In_dict, need_list, self.cfig['norm_oar'], OAR_DICT)
        cat_ptv = In_dict[self.scale_dose_Dict[PatientID]["PTV_High"]['OPTName']]

        data_dict = dict()  
        data_dict['ct_ten'] = In_dict['img']
        data_dict['ct_ten'] = (data_dict['ct_ten'] - data_dict['ct_ten'].mean()) / (data_dict['ct_ten'].std() + 1e-6)
        data_dict['seg_ten'] = torch.cat([cat_ptv, cat_oar], axis=0)
        data_dict['ptv_total_ten'] = In_dict['PTV_Total']
        if 'dose' in In_dict.keys():
            data_dict['dose_ten'] = In_dict['dose'] * In_dict['Body'] 
        data_dict['sample_id'] = ID
        data_dict['params'] = params

        del In_dict
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to(self.dtype)

        return data_dict
