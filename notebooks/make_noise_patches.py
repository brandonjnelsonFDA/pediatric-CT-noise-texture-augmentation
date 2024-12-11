# %% markdown
# The purpose of this notebook is to make noise patches from different sized FOVS and measure their changing NPS
# %%
from pathlib import Path
import numpy as np
from itertools import combinations
from argparse import ArgumentParser

import SimpleITK as sitk
import pydicom
import pandas as pd
from sklearn.feature_extraction.image import PatchExtractor
from tqdm import tqdm

import matplotlib.pyplot as plt


def load_mhd(mhd_file): return sitk.GetArrayFromImage(sitk.ReadImage(mhd_file))


def write_mhd(array, fname): sitk.WriteImage(sitk.GetImageFromArray(array), fname)


def make_noise_images(sa_images, max_images = 500):
    noise_images = []
    for count, image_idxs in enumerate(combinations(range(len(sa_images)), 2)):
        if count < max_images:
            noise_image = sa_images[image_idxs[1]] - sa_images[image_idxs[0]]
            if noise_image.mean() > 100:
                raise RuntimeError(f'Error in noise image at indices {image_idxs}. Mean: {noise_image.mean} > 100')
            noise_images.append(noise_image)
    noise_images = np.array(noise_images)
    noise_images.reshape([*noise_images.shape, 1])
    return noise_images


def make_noise_patches(noise_images, patch_size=(30, 30), max_patches=30):
    return PatchExtractor(patch_size=patch_size, max_patches=max_patches).transform(noise_images)


def load_img(dcm_file):
    dcm = pydicom.read_file(dcm_file)
    return dcm.pixel_array + dcm.RescaleIntercept


def make_noise_image_dict(meta, dose=100, max_images=1000, kernel='fbp'):
    diameters = sorted(meta[(meta.recon==kernel) & (meta.phantom == 'uniform')]['effective diameter [cm]'].unique())
    sa_image_dict = dict()
    for diameter in diameters:
        sa_image_dict[diameter] = np.stack([load_img(fname) for fname in meta[(meta.recon==kernel) &
                                                                              (meta.phantom == 'uniform') &
                                                                              (meta['Dose [%]']==dose) &
                                                                              (meta['effective diameter [cm]'] == diameter)].file])
    noise_image_dict = {k: make_noise_images(v, max_images=max_images) for k, v in sa_image_dict.items()}
    return noise_image_dict


def prep_patches(meta, dose=100, patch_size=(30,30), patches_per_image=30, max_images=1000):
    noise_image_dict = make_noise_image_dict(meta, dose=dose, max_images=max_images)
    print('extracting noise patches...')
    noise_patch_dict = {k: make_noise_patches(v, patch_size, max_patches=patches_per_image) for k, v in tqdm(noise_image_dict.items())}
    return noise_patch_dict


def save_patches(noise_patch_dir, noise_patch_dict, dtype='int16'):
    noise_patch_dir.mkdir(exist_ok=True)
    output_files = []
    for k,v in noise_patch_dict.items():
        outfile = noise_patch_dir / f'diameter{int(k*10):03d}mm.npy'
        print(f'saving patches to {outfile}')
        np.save(outfile, v.astype(dtype))
        output_files.append(outfile)
    return output_files

def get_square_patch(img, center, patch_width=30):
    if img.ndim == 2: img = img[None, :, :]
    return img[:, center[0]-patch_width//2:center[0]+patch_width//2, center[1]-patch_width//2:center[1]+patch_width//2]

def get_patches(img, centers, patch_size=30):
    return {center: get_square_patch(img, center, patch_width=patch_size) for center in centers}

def plot_representative_noise_patches(noise_image_dict, patch_size=64):
    centers = [(256, 256), (50, 256), (110, 110)]
    diams = sorted(list(noise_image_dict.keys()))
    corner_patches = [get_patches(noise_image_dict[d], centers=centers, patch_size=patch_size) for d in diams]

    corners= list(corner_patches[0].keys())
    f, axs = plt.subplots(1,1, dpi=300, figsize=(3,2.25))
    region_patches = np.concatenate([np.concatenate([p[c][0] for p in corner_patches], axis=1) for c in corners])
    axs.imshow(region_patches, cmap='gray')
    axs.set_title(f'{patch_size}x{patch_size} patch images')
    axs.set_xlabel(f'{diams} mm')
    axs.set_ylabel(f'[upper left, top, center]')
    axs.set_xticks([])
    axs.set_yticks([])
    return dict(zip(diams, corner_patches)), corners
# %%
if __name__ == '__main__':
    parser = ArgumentParser(description='Makes noise patches')
    parser.add_argument('--data_path', type=str, default='data', help='directory containing images to be processed')
    parser.add_argument('--save_path', type=str, default='noise_patches', help='save directory for noise patches')
    parser.add_argument('--patch_size', type=int, default=30, help='side length of square patches to be extracted, e.g. patch_size=30 yields 30x30 patches')
    args = parser.parse_args()

    datadir = Path(args.data_path)
    meta = pd.read_csv(datadir / 'metadata.csv')
    meta.file = meta.file.apply(lambda o: datadir / o)
    patch_size = args.patch_size
    noise_patch_dir = Path(args.save_path) / f'patch_size_{patch_size}x{patch_size}'
    if not noise_patch_dir.exists():
        print(f'creating directory: {noise_patch_dir}')
        noise_patch_dir.mkdir(exist_ok=True, parents=True)

    noise_patch_dict = prep_patches(meta, patch_size=(patch_size, patch_size), max_images=500)
    save_patches(noise_patch_dir, noise_patch_dict)