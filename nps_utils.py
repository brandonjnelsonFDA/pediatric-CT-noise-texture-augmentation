from pathlib import Path

import numpy as np
import pandas as pd

from make_noise_patches import make_noise_images, load_mhd


def compute_nps(image):

  if image.ndim == 2:
    image = image[None, :, :]

  if image.ndim == 4:
    image = image[:,:,:,0]

  nsize = image.shape
  nrealization = nsize[0]
  if image.ndim == 3:
    nps = np.zeros((nsize[1],nsize[2]))
    for i in range(nrealization):
      s = np.fft.fftshift(np.fft.fft2(image[i]))
      nps = np.abs(s)**2 + nps
    nps = nps/(nsize[1]*nsize[2])
  else:
    raise ValueError(f'Image of dimension {image.ndim} Not implemented!')
  return nps


def radial_profile(data, center=None):
    center = center or (data.shape[0]/2, data.shape[1]/2)
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile


def get_mean_nps(profile, freq=None):
    profile /= profile.sum()
    freq = freq or list(range(len(profile)))
    return np.sum(np.dot(freq,profile))

def get_info(sa_file):
    dose = int(sa_file.parents[1].stem.split('_')[1])
    recon = sa_file.parents[2].stem
    diameter = int(sa_file.parents[3].stem.split('diameter')[1].split('mm')[0])
    return pd.DataFrame({'diameter [mm]': [diameter], 'dose [%]': [dose], 'recon': [recon], 'filename': [sa_file]})


def make_delta_df(noise_df, measurement='std', ref_recon='fbp'):
    experiment_df = noise_df[noise_df.recon != ref_recon]
    control_df = noise_df[noise_df.recon == ref_recon]

    nrecons = len(experiment_df.recon.unique())
    temp_control = pd.concat(nrecons*[control_df]).reset_index()
    temp_experiment = experiment_df.reset_index()
    col_name = f'$\Delta$ {measurement}'
    unit = 'HU' if measurement == 'std' else '1/pix'
    temp_experiment[f'{col_name} [{unit}]'] = temp_experiment[measurement] - temp_control[measurement]
    temp_experiment[f'{col_name} [%]'] = 100*(temp_control[measurement] - temp_experiment[measurement]) / temp_control[measurement]
    return temp_experiment


def make_results_dict(summary, max_images=2000, verbose=True, diameters=None, doses=None, recons=None):
    """Makes images based measures of nps, std, and keeping select images for later plotting and saves them as a dict for quick access later

    Args:
      summary (pd.DataFrame): DataFrame containing columns for diameter, dose, and recons as well as filenames pointing to raw image files.
      max_images (int):  upper bound on the image combinations made for generating `noise_images`
      verbose (bool): whether to print progress results

    Returns:
      dict
    """
    diameters = diameters or sorted(summary['diameter [mm]'].unique())
    doses = doses or sorted(summary[summary['diameter [mm]']!=200]['dose [%]'].unique())
    recons = recons or summary.recon.unique()

    N = len(diameters)*len(doses)*len(recons)
    idx = 0
    results_dict = dict()
    for diameter in diameters:
        results_dict[diameter] = dict()
        for dose in doses:
            results_dict[diameter][dose] = dict()
            for recon in recons:
                idx += 1
                f = summary[(summary['diameter [mm]'] == diameter) & (summary.recon == recon) & (summary['dose [%]']==dose)].filename
                if verbose & (idx % 10 == 0): print(f'[{idx:03d}/{N:03d}] Making NPS and noise measures on: {diameter}mm, {dose} dose, {recon}')
                vol = load_mhd(f)
                vol = np.squeeze(vol).astype('int16')
                results_dict[diameter][dose][recon] = dict()
                noise_images = make_noise_images(vol, max_images=max_images)
                nps = compute_nps(noise_images)
                std = noise_images.std(axis=(1,2)) 
                nps_profile = radial_profile(nps)
                results_dict[diameter][dose][recon]['image'] = np.copy(vol[0])
                results_dict[diameter][dose][recon]['noise image'] = np.copy(noise_images[0]) #<- important to make an explicit copy of this view or else numpy keeps the 2000x512x512 array for every dose, recon combo and we quickly run out of memory and crash!
                results_dict[diameter][dose][recon]['nps'] = nps
                results_dict[diameter][dose][recon]['profile'] = nps_profile
                results_dict[diameter][dose][recon]['std'] = std
    return results_dict


def get_summary(datadir):
    datadir = Path(datadir)
    sa_filenames = list(datadir.rglob('signal_absent.mhd'))
    return pd.concat([get_info(f) for f in sa_filenames], ignore_index=True)


def append_mean_nps_to_summary_dataframe(results_dict, summary):
    for diam in results_dict.keys():
        for dose in results_dict[diam].keys():
            for recon in results_dict[diam][dose].keys():
                profile = results_dict[diam][dose][recon]['profile']
                mean_nps = get_mean_nps(profile)
                summary.loc[(summary['diameter [mm]'] == diam) & (summary['dose [%]'] == dose) &(summary.recon==recon), 'Mean NPS'] = mean_nps
    return summary


def make_noise_dataframe(results_dict):
    recon = []
    diameter = []
    dose = []
    std = []
    for d in results_dict.keys():
        for dx in results_dict[d].keys():
            for r in results_dict[d][dx].keys():
                for s in results_dict[d][dx][r]['std']:
                    diameter.append(d)
                    dose.append(dx)
                    recon.append(r)
                    std.append(s)
    return pd.DataFrame({'diameter [mm]': diameter, 'dose [%]': dose, 'recon': recon, 'std': std}).sort_values(by=['recon', 'diameter [mm]', 'dose [%]'])


def append_mean_std_to_summary_dataframe(results_dict, summary):
    noise_df = make_noise_dataframe(results_dict)
    merged_noise = noise_df.groupby(['diameter [mm]', 'dose [%]', 'recon']).mean()
    for d in summary['diameter [mm]'].unique():
        for dx in summary['dose [%]'].unique():
            for r in summary['recon'].unique():
                if dx in noise_df['dose [%]'].unique():
                    summary.loc[(summary['diameter [mm]']==d) & (summary['dose [%]']==dx) & (summary['recon']==r), 'mean std'] = merged_noise['std'][d, dx, r]
    return summary
