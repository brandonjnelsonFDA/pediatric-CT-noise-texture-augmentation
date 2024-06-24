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

from scipy import interpolate, signal
from scipy.optimize import curve_fit

def distance_to_spatial_frequencies(distance_array):
    delta_freq = 1/distance_array.max()
    max_freq = 1/np.diff(distance_array)[0]
    sampled_freq = np.arange(0, max_freq, delta_freq)
    return sampled_freq

def sigmoid(x, a, b, c, d):
    return a + (b - a)/(1+10**(c-x)*d)

def measure_mtf(esf, distance=None, oversampling=75, sigmoid_fit=False):
    """
    Compute the modulation transfer function.
    
    Computed the MTF from an edge spread function. Implementation adapted from Friedman et al 2013

    Parameters
    ----------
    esf : array_like
    edge spread function (esf)
    
    distance : array_like, optional
    distance array (pixels, mm, cm, ...) of distances from start to end of the esf
    
    method : str, optional
    method of fitting mtf, options: ['sigmoid fit', 'Han window']

    Friedman SN, Fung GSK, Siewerdsen JH, Tsui BMW.
    A simple approach to measure computed tomography (CT) modulation transfer function (MTF) and noise-power spectrum (NPS) using the American College of Radiology (ACR) accreditation phantom. [Medical Physics. 2013;40(5):051907.
    doi:10.1118/1.4800795](https://onlinelibrary.wiley.com/doi/abs/10.1118/1.4800795)
    """
    
    # interpolate esf
    if distance is None:
        distance = np.array(list(range(len(esf))))
    pixel_size = np.diff(distance)[0]
    
    interpolator = interpolate.interp1d(distance, esf)
    oversampled_distance = np.linspace(distance.min(), distance.max(), num=len(distance)*oversampling) #step 5 Friedman 2013
    oversampled_esf = interpolator(oversampled_distance)
    
    if sigmoid_fit:
        popt, pcov = curve_fit(sigmoid, oversampled_distance, oversampled_esf)
        oversampled_esf = sigmoid(oversampled_distance, *popt)
        
    # normalize esf to within [0, 1] with transition
    rel_dist_from_ends_for_means = 0.25 # larger number includes more values but risks getting to close to the edge
    mean_signal = oversampled_esf[:round(len(oversampled_distance)*rel_dist_from_ends_for_means)].mean()
    mean_bkg = oversampled_esf[round(-len(oversampled_esf)*rel_dist_from_ends_for_means):-1].mean()
    
    oversampled_esf = (oversampled_esf - oversampled_esf.min())/(oversampled_esf.max() - oversampled_esf.min()) #eq. 4 Friedman 2013
    
    # ensures signal goes from low signal to high signal for positive derivative when calculating line spread function
    if mean_bkg < mean_signal:
        oversampled_esf = 1 - oversampled_esf

    lsf = np.diff(oversampled_esf) # ESF --> LSF step 6

    rel_hann_width = 0.03
    hann_width = round(len(lsf)*rel_hann_width)
    win = signal.windows.hann(hann_width)
    lsf = signal.convolve(lsf, win, mode='same') / sum(win)  # hann windowing step 7
    lsf *= signal.windows.hann(len(lsf))**4
    lsf[:round(len(oversampled_distance)*rel_dist_from_ends_for_means)]=0
    lsf[-round(len(oversampled_distance)*rel_dist_from_ends_for_means):]=0

    # calculate mtf as modulus of fft
    mtf = np.abs(np.fft.fft(lsf))
    # mtf = np.abs(np.fft.fft(filtered_lsf))
    freq = distance_to_spatial_frequencies(oversampled_distance)
    cutoff_freq = 1/pixel_size/2
    cutoff_freq_idx=np.argwhere(freq >= cutoff_freq).squeeze()[0]
    
    return mtf[:cutoff_freq_idx], freq[:cutoff_freq_idx]

from tqdm import tqdm
import pandas as pd

def make_mtf_df(avg_esf, sigmoid_fit=False):
    '''
    method: ['sigmoid fit', 'han window']
    '''
    diameters = []
    fovs = []
    doses = []
    recons = []
    contrasts = []
    frequencies_invcm = []
    frequencies_invpix = []
    mtfs = []
    names = []

    for diameter in tqdm(avg_esf['effective diameter [cm]'].unique()):
        temp = avg_esf[avg_esf['effective diameter [cm]']==diameter]
        for fov in temp['FOV [cm]'].unique():
            temp = temp[temp['FOV [cm]']==fov].copy()
            for dose in temp['Dose [%]'].unique():
                for recon in temp['recon'].unique():
                    for contrast in temp['contrast'].unique():
                        esf = temp[(temp['FOV [cm]']==fov)&(temp['Dose [%]']==dose)&(temp['recon']==recon)&(temp['contrast']==contrast)].drop_duplicates()
                        distance_px = esf['distance'].to_numpy()
                        distance_cm = esf['distance [cm]'].to_numpy()
                        esf_profile = esf['ESF'].to_numpy()
                        assert(len(distance_px) == len(distance_cm))
                        assert(len(np.unique(distance_px)) == len(distance_px))

                        mtf, freq_invcm = measure_mtf(esf_profile, distance_cm, sigmoid_fit=sigmoid_fit)
                        mtf_px, freq_invpx = measure_mtf(esf_profile, distance_px, sigmoid_fit=sigmoid_fit)
                        if len(freq_invcm) < len(freq_invpx):
                            freq_invcm = np.append(freq_invcm, freq_invcm[-1]+np.diff(freq_invcm)[0])
                            mtf = np.append(mtf, 0)
                        elif len(freq_invcm) > len(freq_invpx):
                            freq_invpx = np.append(freq_invpx, freq_invpx[-1]+np.diff(freq_invpx)[0])
                            mtf_px = np.append(mtf_px, 0)
                        assert(len(freq_invcm) == len(freq_invpx))
                        name = f'{diameter} cm {dose}% dose {recon} {contrast} HU'
                        names += len(mtf)*[name]
                        diameters += len(mtf)*[diameter]
                        fovs += len(mtf)*[fov]
                        doses += len(mtf)*[dose]
                        recons += len(mtf)*[recon]
                        contrasts += len(mtf)*[contrast]
                        frequencies_invcm += list(freq_invcm)
                        frequencies_invpix += list(freq_invpx)
                        mtfs += list(mtf)
    mtf_df = pd.DataFrame({'name': names,
                           'effective diameter [cm]': diameters,
                           'FOV [cm]': fovs,
                           'Dose [%]': doses,
                           'recon': recons,
                           'contrast': contrasts,
                           'spatial frequency [1/cm]': frequencies_invcm,
                           'spatial frequency [1/px]': frequencies_invpix,
                           'MTF': mtfs})
    return mtf_df