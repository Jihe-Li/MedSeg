import pandas as pd
import SimpleITK as sitk
import os
import numpy as np
from tqdm import tqdm


read_root = '/storage/research/artorg_mia/Head_and_Neck/HECKTOR2025_resampled_iso'
df = pd.read_csv('data/HECKTOR/meta_data_hecktor.csv')
df = df.loc[df['CT'] == True]
df = df.loc[df['PT'] == True]
df = df.loc[df['GTV'] == True]
hecktor_list = df['PID'].tolist()

no_GTV_list = []
no_nodule_list = []
emplty_list = []
for PatientID in tqdm(hecktor_list, desc='Resampling', ncols=50):
    gtv_path = os.path.join(read_root, PatientID, f'{PatientID}.nii.gz')
    gtv_img = sitk.ReadImage(gtv_path)
    gtv_arr = sitk.GetArrayFromImage(gtv_img)
    if np.sum(gtv_arr == 1) == 0:
        no_GTV_list.append(PatientID)
    if np.sum(gtv_arr == 2) == 0:
        no_nodule_list.append(PatientID)
    if np.sum(gtv_arr) == 0:
        emplty_list.append(PatientID)

print(len(no_GTV_list))
print(len(no_nodule_list))
print(len(emplty_list))
