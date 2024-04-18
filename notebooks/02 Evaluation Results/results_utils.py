import sys
sys.path.append('..')

from ipywidgets import interact, Checkbox
import numpy as np
import matplotlib.pyplot as plt

from utils import center_crop, center_crop_like, load_mhd, get_ground_truth, ctshow

def browse_studies(meta, phantom='anthropomorphic', diameter=16, fov=20.8, dose=100, recon='fbp', crop=False):
    phantom_df =  meta[(meta.phantom==phantom)]
    if diameter not in phantom_df['effective diameter [cm]'].unique():
        print(f"diameter {diameter} not in {phantom_df['effective diameter [cm]'].unique()}")
        return
    phantom_df = phantom_df[(phantom_df.phantom==phantom) & (phantom_df['effective diameter [cm]']==diameter)]
    available_fovs = sorted(phantom_df['FOV [cm]'].unique())
    if fov not in available_fovs:
        print(f'FOV {fov} not in {available_fovs}')
        return
    available_doses = sorted(phantom_df['Dose [%]'].unique())
    if dose not in available_doses:
        print(f'dose {dose}% not in {available_doses}')
        return
    patient = phantom_df[(phantom_df.phantom==phantom) &
                         (phantom_df['Dose [%]']==dose) &
                         (phantom_df['effective diameter [cm]'] == diameter) &
                         (phantom_df['FOV [cm]'] == fov) &
                         (phantom_df['recon'] == recon)].iloc[0]
    img = load_mhd(patient.file)[0]
    gt = load_mhd(get_ground_truth(patient.file))
    if crop:
        img = center_crop_like(img, gt)
        gt = center_crop(gt)
    ctshow(np.concatenate([img, gt], axis=1))
    plt.title(f"{patient['Dose [%]']}% dose {patient.recon} | ground truth")
    

def study_viewer(meta): 
    viewer = lambda phantom='anthropmorphic', dose=100, diameter=16, fov=20.8, recon='fbp', crop=False: browse_studies(meta, phantom=phantom, dose=dose, diameter=diameter, fov=fov, recon=recon, crop=crop)
 
    interact(viewer,
             phantom=meta.phantom.unique(),
             dose=sorted(meta['Dose [%]'].unique(), reverse=True),
             diameter = sorted(meta['effective diameter [cm]'].unique()),
             fov=sorted(meta['FOV [cm]'].unique()),
             recon=meta['recon'].unique(),
             crop=Checkbox(value=False, description='crop image'))