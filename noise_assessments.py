# %%
from pathlib import Path
import pickle
from argparse import ArgumentParser

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from nps_utils import make_delta_df, make_results_dict, get_summary,\
     append_mean_nps_to_summary_dataframe, make_noise_dataframe, append_mean_std_to_summary_dataframe

sns.set_theme()


def plot_images(results_dict, results_dir=None, diameters=[112, 151, 216, 350]):
    recons = results_dict[112][100].keys()
    def normalize(x, recon):
        if recon == 'fbp':
            return x-1000
        else:
            return x
    images = np.concatenate([np.concatenate([normalize(results_dict[d][100][r]['image'], r) for d in diameters], axis=1) for r in recons], axis=0)
    mean = 0
    std = results_dict[112][100]['fbp']['noise image'].std()
    N = 1
    f, ax = plt.subplots(figsize=(6.5, 4))
    ax.imshow(images, vmin=mean-N*std, vmax=mean+N*std, cmap='gray')
    ax.grid(False)
    ax.set_xlabel(f'{diameters} mm diameters')
    ax.set_ylabel(['FBP', 'CNN', 'Augmented'][::-1])
    if results_dir is None: return
    outfilename = Path(results_dir)/'images.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_noise_images(results_dict, results_dir=None, diameters=[112, 151, 216, 350]):
    recons = list(results_dict[112][100].keys())
    noise_images = np.concatenate([np.concatenate([results_dict[d][100][r]['noise image'] for d in diameters], axis=1) for r in recons], axis=0)
    std = noise_images.std()
    N=1
    f, ax = plt.subplots(figsize=(6.5, 4))
    ax.imshow(noise_images, cmap='gray',vmin=-N*std, vmax=N*std)
    ax.grid(False)
    ax.set_ylabel(['FBP', 'CNN', 'Augmented'][::-1])
    ax.set_xlabel(f'{diameters} mm diameters')
    if results_dir is None: return
    f.tight_layout()
    outfilename = Path(results_dir)/'noise_images.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_nps_images(results_dict, results_dir=None, diameters = [112, 151, 216, 350]):
    results_dir = results_dir or '.'
    results_dir = Path(results_dir)
    recons = list(results_dict[112][100].keys())
    nps_images = np.concatenate([np.concatenate([results_dict[d][100][r]['nps'] for d in diameters], axis=1) for r in recons], axis=0)
    f, ax = plt.subplots(figsize=(6.5, 4))
    ax.imshow(nps_images)
    ax.grid(False)
    ax.set_ylabel(['FBP', 'CNN', 'Augmented'][::-1])
    ax.set_xlabel(f'{diameters} mm diameters')
    if results_dir is None: return
    outfilename = Path(results_dir)/'nps_images.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_nps_profiles(results_dict, results_dir=None, diameters=[112, 151, 292]):
    colors = ['black', 'red', 'blue']
    f, ax = plt.subplots(figsize=(4.5, 4))
    for c, d in zip(colors, diameters):
        h0, = ax.plot(results_dict[d][100]['fbp']['profile'], label=f'{d} fbp', linestyle='-', color=c)
        h1, = ax.plot(results_dict[d][100]['simple CNN MSE']['profile'], label=f'{d} CNN', linestyle='--', color=c)
        h2, = ax.plot(results_dict[d][100]['simple CNN MSE with augmentation']['profile'], label=f'{d} augment', linestyle=':', color=c)
        if c=='black':     
            ax.legend([h0, h1, h2], ['FBP', 'CNN', 'Augmented'])
    ax.annotate("112 mm", (75, 1.2*1e7))
    ax.annotate("151 mm", (120, 0.7*1e7), color='red')
    ax.annotate("292 mm", (300, 0.3*1e7), color='blue')

    ax.set_ylabel('Noise Power')
    ax.set_xlabel('Spatial Frequency [1/pix]')
    if results_dir is None: return
    outfilename = Path(results_dir)/'nps.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_mean_nps(summary, results_dir=None, dose=[100]):
    data = summary[(summary['diameter [mm]'] != 200) & (summary['dose [%]'].isin(dose))].copy()
    delta_df = make_delta_df(data, measurement='Mean NPS')
    delta_df['$\Delta$ Mean NPS [1/pix]'] = -1*delta_df['$\Delta$ Mean NPS [1/pix]']
    f, axs = plt.subplots(1,2, figsize=(9, 4))
    sns.lineplot(ax=axs[0], data=data, x='diameter [mm]', y='Mean NPS', style='recon', hue='dose [%]')
    sns.lineplot(ax=axs[1], data=delta_df, x='diameter [mm]', y='$\Delta$ Mean NPS [1/pix]', style='recon', hue='dose [%]')
    axs[0].set_ylabel('Mean NPS [1/pix]')
    axs[1].set_ylabel(r'''$\Delta$Mean NPS [1/pix]
($\overline{NPS}-\overline{NPS}_{FBP}$)''')
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.35,1.1), ncol=1, fontsize=8)
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.35,1.1), ncol=1, fontsize=8)
    f.tight_layout()
    if results_dir is None: return
    outfilename = Path(results_dir)/'mean_nps.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_noise_v_diameter(results_dict, results_dir=None, doses=[100, 25]):
    noise_df = make_noise_dataframe(results_dict)
    noise_df = noise_df[noise_df['diameter [mm]'] != 200]
    doses = [100, 25]
    noise_df = noise_df[noise_df['dose [%]'].isin(doses)]
    delta_df = make_delta_df(noise_df, measurement='std')
    # delta_df['$\Delta$ std'] = -1*delta_df['$\Delta$ std']
    f, axs = plt.subplots(1,2, figsize=(8, 4))
    sns.lineplot(ax=axs[0], data=noise_df, x='diameter [mm]', y='std', hue='dose [%]', style='recon')
    axs[0].set_ylabel('std [HU]')
    sns.lineplot(ax=axs[1], data=delta_df, x='diameter [mm]', y='$\Delta$ std [HU]', style='recon', hue='dose [%]')
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.4), ncol=2, fontsize=8)
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5,1.4), ncol=2, fontsize=8)
    f.tight_layout()
    if results_dir is None: return
    outfilename = Path(results_dir)/'std_noise.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def plot_noise_reduction(results_dict, results_dir=None, doses=[100, 25]):
    noise_df = make_noise_dataframe(results_dict)
    noise_df = noise_df[noise_df['diameter [mm]'] != 200]
    noise_df = noise_df[noise_df['dose [%]'].isin(doses)]
    delta_df = make_delta_df(noise_df, 'std')

    f, ax = plt.subplots(figsize=(4.5, 4))
    sns.lineplot(ax=ax, data=delta_df, x='diameter [mm]', y='$\Delta$ std [%]', style='recon', hue='dose [%]')
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.3), ncol=2)
    f.tight_layout()
    if results_dir is None: return
    outfilename = Path(results_dir)/'noise_reduction.png'
    f.savefig(outfilename, dpi=600, bbox_inches='tight')
    return outfilename


def load_data(datadir, results_dir):
    datadir = Path(datadir)
    results_dir = Path(results_dir)

    summary = get_summary(datadir)
    summary.to_csv(datadir/'summary.csv', index=False)

    intermediate_results = results_dir/'results.pkl'
    if intermediate_results.exists():
        with open(intermediate_results, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        print(f'Making measurements and caching intermediate results to {intermediate_results} for plotting, this only needs to be done once per experiment...')
        results_dict = make_results_dict(summary, doses=[100, 25])
        with open(intermediate_results, 'wb') as f:
            pickle.dump(results_dict, f)

    summary = append_mean_nps_to_summary_dataframe(results_dict, summary)
    summary = append_mean_std_to_summary_dataframe(results_dict, summary)
    summary.to_csv(results_dir / 'summary.csv')
    return results_dict, summary


def main(datadir, results_dir='results/test'):
    
    results_dict, summary = load_data(datadir, results_dir)
    
    out = plot_images(results_dict, results_dir, diameters=[112, 151, 216, 350])
    print(f'results saved to: {out}')

    out = plot_noise_images(results_dict, results_dir, diameters=[112, 151, 216, 350])
    print(f'results saved to: {out}')

    out = plot_nps_images(results_dict, results_dir, diameters=[112, 151, 216, 350])
    print(f'results saved to: {out}')

    out = plot_nps_profiles(results_dict, results_dir, diameters=[112, 151, 292])
    print(f'results saved to: {out}')

    out = plot_mean_nps(summary, results_dir, dose=[100])
    print(f'results saved to: {out}')

    out = plot_noise_v_diameter(results_dict, results_dir, doses=[100, 25])
    print(f'results saved to: {out}')

    out = plot_noise_reduction(results_dict, results_dir, doses=[100, 25])
    print(f'results saved to: {out}')


datadir = '/gpfs_projects/brandon.nelson/PediatricCTSizeDataAugmentation/CCT189_peds'
results_dir = 'results/test'
# %%

if __name__ == '__main__':

    parser = ArgumentParser(description='Make Image Quality Summary Plots')
    parser.add_argument('base_directory', nargs='?', default="", help='directory containing images to be processed')
    parser.add_argument('-o', '--output_directory', type=str, required=False, default="results/test", help='directory to save resulting plots and files')

    args = parser.parse_args()

    datadir = args.base_directory or datadir
    results_dir = args.output_directory or results_dir

    main(datadir=Path(datadir), results_dir=Path(results_dir))

# %%
def get_square_patch(img, center, patch_width=30):
    if img.ndim == 2: img = img[None, :, :]
    return img[:, center[0]-patch_width//2:center[0]+patch_width//2, center[1]-patch_width//2:center[1]+patch_width//2]
# %%
def get_patches(img, centers, patch_size=30):
    return {center: get_square_patch(img, center, patch_width=patch_size) for center in centers}
# %%
results_dict, summary = load_data(datadir, results_dir)
# %%
patches = get_patches(results_dict[112][100]['fbp']['image'], [(110, 110)])
plt.imshow(list(patches.values())[0][0])
# %%
diams = [112, 151, 216, 350]
f, axs = plt.subplots(1, 1, figsize=(6,1.5), dpi=300)
images = np.concatenate([results_dict[d][100]['fbp']['image']-1000 for d in diams], axis=1)
coords = [(256, 256), (50, 256), (110, 110)]
image_patches = [get_patches(results_dict[d][100]['fbp']['image']-1000, centers=coords) for d in diams]

image_stds = [{k: (img.mean(), img.std()) for k, img in p.items()} for p in image_patches]
ww = 40
wl = 0
N=1
axs.imshow(images, cmap='gray', vmin=wl-ww//2, vmax=wl+ww//2)
axs.axis('off')
axs.set_title('images')

for idx, p in enumerate(image_stds):
    for xy, (mean, std) in p.items():
        axs.annotate(f'[{mean:2.1f}, {std:2.1f}] HU', (xy[0] + idx*512, xy[1]), fontsize=4, bbox=dict(fc="lightblue", ec="steelblue"))
# %%
diams = [112, 151, 216, 350]
f, axs = plt.subplots(1, 1, figsize=(6,1.5), dpi=300)
images = np.concatenate([results_dict[d][100]['fbp']['noise image'] for d in diams], axis=1)
image_patches = [get_patches(results_dict[d][100]['fbp']['noise image'], centers=coords) for d in diams]

image_stds = [{k: (img.mean(), img.std()) for k, img in p.items()} for p in image_patches]
ww2 = np.sqrt(2*ww**2)
N=1
axs.imshow(images, cmap='gray', vmin=wl-ww2//2, vmax=wl+ww2//2)
axs.axis('off')
axs.set_title('noise images')

for idx, p in enumerate(image_stds):
    for xy, (mean, std) in p.items():
        axs.annotate(f'[{mean:2.1f}, {std:2.1f}] HU', (xy[0] + idx*512, xy[1]), fontsize=4, bbox=dict(fc="lightblue", ec="steelblue"))
# %%
patches = [get_corner_patches(results_dict[d][100]['fbp']['noise image']) for d in diams]

images = np.concatenate([np.concatenate([patches[idx][c][0] for idx, d in enumerate(diams)], axis=1) for c in patches[0].keys()], axis=0)
mean = images.mean()
std = images.std()
N=1

axs[2].imshow(images, cmap='gray', vmin=mean-N*std, vmax=mean+N*std)
axs[2].axis('off')
axs[2].set_title('patches')

images = np.concatenate([np.concatenate([patches[idx][c][0] for idx, d in enumerate(diams)], axis=1) for c in patches[0].keys()], axis=0)
# see make_noise_patches.ipynb for this one
axs[3].set_title('patch nps')

print('rather than subplots, consider saving out as equal sized images and having latex make the subplots')
# %%