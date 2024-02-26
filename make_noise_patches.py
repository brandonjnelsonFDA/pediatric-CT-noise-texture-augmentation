# %% markdown
# The purpose of this notebook is to make noise patches from different sized FOVS and measure their changing NPS
# %%
from pathlib import Path
import numpy as np
from itertools import combinations
from argparse import ArgumentParser

import SimpleITK as sitk
from sklearn.feature_extraction.image import PatchExtractor
from tqdm import tqdm


def load_mhd(mhd_file): return sitk.GetArrayFromImage(sitk.ReadImage(mhd_file))


def write_mhd(array, fname): sitk.WriteImage(sitk.GetImageFromArray(array), fname)


def make_noise_images(sa_images, max_images = 500):
    noise_images = []
    for idx, s in enumerate(combinations(sa_images, 2)):
        if idx < max_images:
            noise_images.append(s[1] - s[0])
    noise_images = np.array(noise_images)
    noise_images.reshape([*noise_images.shape, 1])
    return noise_images


def make_noise_patches(noise_images, patch_size=(30, 30), max_patches=30):
    return PatchExtractor(patch_size=patch_size, max_patches=max_patches).transform(noise_images)


def make_noise_image_dict(datadir, dose=100, max_images=1000, kernel='fbp'):
    datadir = Path(datadir)

    sa_file_dict={d.stem : d/f'{kernel}/dose_{dose:03d}/signal_absent/signal_absent.mhd' for d in datadir.glob('diameter*mm')}
    print(f'generating {max_images} {kernel} noise images from the following phantom scans: {sorted(sa_file_dict.keys())}')
    sa_image_dict = {k: load_mhd(v)-1000 for k, v in tqdm(sa_file_dict.items())}
    noise_image_dict = {k: make_noise_images(v, max_images=max_images) for k, v in sa_image_dict.items()}
    return noise_image_dict


def prep_patches(datadir, dose=100, patch_size=(30,30)):
    noise_image_dict = make_noise_image_dict(datadir, dose=dose)
    print('extracting noise patches...')
    noise_patch_dict = {k: make_noise_patches(v, patch_size) for k, v in tqdm(noise_image_dict.items())}
    return noise_patch_dict


def save_patches(noise_patch_dir, noise_patch_dict, dtype='int16'):
    noise_patch_dir.mkdir(exist_ok=True)
    output_files = []
    for k,v in noise_patch_dict.items():
        outfile = noise_patch_dir / f'{k}.npy'
        print(f'saving patches to {outfile}')
        np.save(outfile, v.astype(dtype))
        output_files.append(outfile)
    return output_files

# %%
if __name__ == '__main__':
    parser = ArgumentParser(description='Makes noise patches')
    parser.add_argument('--data_path', type=str, default='data', help='directory containing images to be processed')
    parser.add_argument('--save_path', type=str, default='noise_patches', help='save directory for noise patches')
    parser.add_argument('--patch_size', type=int, default=30, help='side length of square patches to be extracted, e.g. patch_size=30 yields 30x30 patches')
    args = parser.parse_args()

    datadir = args.data_path
    patch_size = args.patch_size
    noise_patch_dir = Path(args.save_path) / f'patch_size_{patch_size}x{patch_size}'
    if not noise_patch_dir.exists():
        print(f'creating directory: {noise_patch_dir}')
        noise_patch_dir.mkdir(exist_ok=True, parents=True)

    noise_patch_dict = prep_patches(datadir, patch_size=(patch_size, patch_size))
    save_patches(noise_patch_dir, noise_patch_dict)