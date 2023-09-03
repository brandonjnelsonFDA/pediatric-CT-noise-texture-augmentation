# %% markdown
# The purpose of this notebook is to make noise patches from different sized FOVS and measure their changing NPS
# %%
from pathlib import Path
import numpy as np
from itertools import combinations
from argparse import ArgumentParser

import SimpleITK as sitk
from sklearn.feature_extraction.image import PatchExtractor


def load_mhd(mhd_file): return sitk.GetArrayFromImage(sitk.ReadImage(mhd_file))


def write_mhd(array, fname): sitk.WriteImage(sitk.GetImageFromArray(array), fname)


def make_noise_images(sa_images, max_images = 2000):
    noise_images = []
    for idx, s in enumerate(combinations(sa_images, 2)):
        if idx < max_images:
            noise_images.append(s[1] - s[0])
    noise_images = np.array(noise_images)
    noise_images.reshape([*noise_images.shape, 1])
    return noise_images


def make_noise_patches(noise_images, patch_sz=(30, 30), max_patches=30):
    return PatchExtractor(patch_size=patch_sz, max_patches=max_patches).transform(noise_images)


def make_noise_image_dict(datadir, dose=100):
    datadir = Path(datadir)

    sa_file_dict={d.stem : next(d.rglob(f'dose_{dose:03d}/*/signal_absent.mhd')) for d in datadir.glob('diameter*mm')}
    print(len(sa_file_dict))
    sa_image_dict = {k: load_mhd(v)-1000 for k, v in sa_file_dict.items()}

    noise_image_dict = {k: make_noise_images(v, 2000) for k, v in sa_image_dict.items()}
    return noise_image_dict


def prep_patches(datadir, dose=100, patch_sz=(30,30)):
    noise_image_dict = make_noise_image_dict(datadir, dose=dose)
    noise_patch_dict = {k: make_noise_patches(v, patch_sz) for k, v in noise_image_dict.items()}
    return noise_patch_dict


def save_patches(noise_patch_dir, noise_patch_dict):
    noise_patch_dir.mkdir(exist_ok=True)

    output_files = []
    for k,v in noise_patch_dict.items():
        outfile = noise_patch_dir / f'{k}.npy'
        np.save(outfile, v)
        output_files.append(outfile)
    return output_files

# %%
if __name__ == '__main__':

    parser = ArgumentParser(description='Makes noise patches')
    parser.add_argument('base_directory', type=str, default="", help='directory containing images to be processed')
    args = parser.parse_args()

    datadir = args.base_directory or 'data'

    noise_patch_dir = Path('noise_patches')

    noise_patch_dict = prep_patches(datadir)
    save_patches(noise_patch_dir, noise_patch_dict)