# %%
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nps_utils import compute_nps
from noise_assessments import load_data
from make_noise_patches import make_noise_image_dict

datadir = '/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds'
results_dir = 'results/test'


def get_square_patch(img, center, patch_width=30):
    if img.ndim == 2: img = img[None, :, :]
    return img[:, center[0]-patch_width//2:center[0]+patch_width//2, center[1]-patch_width//2:center[1]+patch_width//2]


def get_patches(img, centers, patch_size=30):
    return {center: get_square_patch(img, center, patch_width=patch_size) for center in centers}


def plot_methods(datadir, results_dir=None):
    datadir = Path(datadir)
    results_dict, summary = load_data(datadir, results_dir)

    noise_image_dict = make_noise_image_dict(Path(datadir) / 'CCT189_peds_fbp')

    diams = [112, 185, 216]

    fig = plt.figure(figsize=(4, 5), dpi=300)
    gs = gridspec.GridSpec(3, 2, wspace=0.15, hspace=0.15)

    images = np.concatenate([results_dict[d][100]['fbp']['image']-1000 for d in diams], axis=1)
    coords = [(256, 256), (50, 256), (110, 110)]
    image_patches = [get_patches(results_dict[d][100]['fbp']['image']-1000, centers=coords) for d in diams]

    image_stds = [{k: (img.mean(), img.std()) for k, img in p.items()} for p in image_patches]
    ww = 40
    wl = 0
    N=1
    ax = fig.add_subplot(gs[0, :])
    ax.imshow(images, cmap='gray', vmin=wl-ww//2, vmax=wl+ww//2)
    ax.axis('off')
    ax.set_title(f'(a) {diams} mm images')

    for idx, p in enumerate(image_stds):
        for xy, (mean, std) in p.items():
            ax.annotate(f'[{mean:2.0f}, {std:2.0f}] HU', (xy[0] + idx*512, xy[1]), fontsize=5, bbox=dict(boxstyle='square,pad=0.3', fc="lightblue", ec="steelblue"))

    images = np.concatenate([results_dict[d][100]['fbp']['noise image'] for d in diams], axis=1)
    image_patches = [get_patches(results_dict[d][100]['fbp']['noise image'], centers=coords) for d in diams]

    image_stds = [{k: (img.mean(), img.std()) for k, img in p.items()} for p in image_patches]
    ww2 = np.sqrt(2*ww**2)
    N=1
    ax = fig.add_subplot(gs[1, :])
    ax.imshow(images, cmap='gray', vmin=wl-ww2//2, vmax=wl+ww2//2)
    ax.axis('off')
    ax.set_title(f'(b) {diams} mm noise images')

    for idx, p in enumerate(image_stds):
        for xy, (mean, std) in p.items():
            ax.annotate(f'[{mean:2.0f}, {std:2.0f}] HU', (xy[0] + idx*512, xy[1]), fontsize=5, bbox=dict(boxstyle='square,pad=0.3', fc="lightblue", ec="steelblue"))

    centers = [(256, 256), (50, 256), (110, 110)]
    corner_patches = [get_patches(noise_image_dict[f'diameter{x}mm'], centers=centers) for x in diams]

    corners= list(corner_patches[0].keys())
    region_patches = np.concatenate([np.concatenate([p[c][0] for p in corner_patches], axis=1) for c in corners])

    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(region_patches, cmap='gray', aspect='equal')
    ax.set_title('(c) patch images')
    ax.set_xlabel(f'{diams} mm', fontsize=8)
    ax.set_ylabel(f'[upper left, top, center]', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


    ax = fig.add_subplot(gs[2, 1])
    ax.imshow(np.concatenate([np.concatenate([compute_nps(p[c]) for p in corner_patches], axis=1) for c in corners]), aspect='equal')
    ax.set_title('(d) patch NPS')
    ax.set_xlabel(f'{diams} mm', fontsize=8)
    ax.set_ylabel(f'[upper left, top, center]', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if results_dir is None: return
    fig.savefig(Path(results_dir) /'methods.png', dpi=600, bbox_inches='tight')

# %%
if __name__ == '__main__':

    parser = ArgumentParser(description='Make Methods Plots')
    parser.add_argument('base_directory', nargs='?', default="", help='directory containing images to be processed')
    parser.add_argument('-o', '--output_directory', type=str, required=False, default="results/test", help='directory to save resulting plots and files')

    args = parser.parse_args()

    datadir = args.base_directory or datadir
    results_dir = args.output_directory or results_dir

    plot_methods(datadir=Path(datadir), results_dir=Path(results_dir))
# %%
