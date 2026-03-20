import os
import torch
import yaml
import json
import numpy as np
import SimpleITK as sitk


def gettenfromimg(img: sitk.Image):
    arr = sitk.GetArrayFromImage(img)
    arr = torch.FloatTensor(arr)
    params = {
        "origin": img.GetOrigin(),
        "spacing": img.GetSpacing(),
        "direction": img.GetDirection(),
        "size": img.GetSize()
    }
    return arr, params

def load_image(path, return_tensor=False):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    if return_tensor:
        arr = torch.FloatTensor(arr)

    params = {
        "origin": img.GetOrigin(),
        "spacing": img.GetSpacing(),
        "direction": img.GetDirection(),
        "size": img.GetSize()
    }

    return arr, params

def save_image(arr, path, params, is_tensor=False, is_label=False):
    if is_tensor:
        if is_label:
            arr = arr.to(torch.uint8)
        else:
            arr = arr.float()
        arr = arr.detach().cpu().numpy()
    img = sitk.GetImageFromArray(arr)
    img.SetOrigin(params['origin'])
    img.SetSpacing(params['spacing'])
    img.SetDirection(params['direction'])
    if len(os.path.dirname(path)) > 0:
        os.makedirs(os.path.dirname(path), exist_ok=True, )
    sitk.WriteImage(img, path)

def load_landmarks(path):
    """Load landmarks from a text file."""
    with open(path) as f:
        landmarks = torch.FloatTensor(
            [list(map(float, line[:-1].split("\t")[:3])) for line in f.readlines()])

    return landmarks

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        # 使用 json.load() 从文件中读取并解析 JSON 数据
        data = json.load(f)
    return data

def save_arr_to_csv(arr, path):
    """
    Save a numpy array to a CSV file.

    Args:
        arr (np.ndarray): The numpy array to save.
        path (str): The file path where the array will be saved as CSV.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, arr, delimiter=",")

def load_csv_to_arr(path):
    """
    Load a CSV file as a numpy array.
    
    Handles "nan" strings in the CSV and converts them to numpy NaN values.
    Empty lines are automatically skipped.

    Args:
        path (str): The file path of the CSV file to load.

    Returns:
        np.ndarray: The loaded data as a numpy array with dtype float64.
                    NaN values are represented as np.nan.
    """
    arr = np.genfromtxt(path, delimiter=",", missing_values="nan", filling_values=np.nan)
    return arr
