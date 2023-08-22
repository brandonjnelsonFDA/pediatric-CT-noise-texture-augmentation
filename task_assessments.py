# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

sns.set_theme()


def clean_dataframe(res):
    res = res[~res.duplicated()].copy()
    res.observer = res.observer.str.strip()
    res.recon = res.recon.str.strip()
    res.rename(columns={'dose_level': 'dose level [%]'}, inplace=True)
    res = res[res.diameter != 200]
    return res


def plot_auc_v_diameter(res, diam, f=None, ax=None):
    if (f is None) and (ax is None):
        f, ax = plt.subplots(figsize = (8,4))
    data = res if diam=='average' else res[(res.diameter==diam)]
    sns.lineplot(ax=ax, data=data, x='dose level [%]', y='auc', hue='recon', style='observer')
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.5), ncol=2)
    ax.set_title(f'AUC vs. dose: {diam} mm')
    return f, ax


def plot_auc_v_dose(res, dose, f=None, ax=None):
    if (f is None) and (ax is None):
        f, ax = plt.subplots(figsize = (8,4))
    data = res if dose=='average' else res[(res['dose level [%]']==dose)]
    sns.lineplot(ax=ax, data=data, x='diameter', y='auc', hue='recon', style='observer')
    ax.legend( loc="upper center", bbox_to_anchor=(0.5,1.5), ncol=2)
    ax.set_title(f'AUC vs. diameter: {dose} % dose level')
    return f, ax


def plot_delta_auc(res, x='diameter', f=None, ax=None):
    if (f is None) and (ax is None):
        f,ax = plt.subplots(figsize= (6, 5))
    ref_auc = res[res.recon == 'fbp'].pop('auc')
    nrecons = len(res[res.recon != 'fbp'].recon.unique())
    ref_auc = np.concatenate([ref_auc for i in range(nrecons)])
    delta_df = res[res.recon != 'fbp'].copy()
    delta_df['$\Delta auc$'] = delta_df['auc'] - ref_auc
    delta_df.pop('auc')
    delta_df.pop('snr')
    sns.lineplot(ax=ax, data=delta_df, x=x, y = '$\Delta auc$', hue='recon', style='observer')
    ax.hlines(y=0, xmin=res[x].min(), xmax=res[x].max(), color='black', linestyle='--', linewidth=3.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.2), ncol=2)
    return f, ax


def main(results_csv, results_dir=None):

    results = Path(results_csv)
    results_dir = results_dir or results.parent
    results_dir = Path(results_dir)
    res = pd.read_csv(results)
    # auc v diameter
    res = clean_dataframe(res)
    for diam in [*res.diameter.unique(), 'average']:
        f, ax = plot_auc_v_diameter(res, diam)
        fname = results_dir / f'auc_v_dose_{diam}mm.png'
        print(f'writing to: {fname}')
        f.savefig(fname, dpi=600, facecolor='white', bbox_inches='tight')

    #auc v dose
    for dose in [*res['dose level [%]'].unique(), 'average']:
        f, ax = plot_auc_v_dose(res, dose)
        dose = dose if dose == 'average' else f'{dose:03d}'
        fname = results_dir / f'auc_v_diameter_{dose}dose.png'
        print(f'writing to: {fname}')
        f.savefig(fname, dpi=600, facecolor='white', bbox_inches='tight')

    #delta auc v diameter
    f, ax = plot_delta_auc(res)
    ax.get_legend().remove()
    ax.set_title('Task advantage following denoising')
    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5,1.2), ncol=2)
    f.tight_layout()
    f.savefig(results_dir/'diffauc_v_diameter.png', dpi=600, facecolor='white', bbox_inches='tight')

    #delta auc v diameter per HU insert
    HUs = res.insert_HU.unique()
    f, axs = plt.subplots(2, 2, sharex=True,sharey=True, tight_layout=True, figsize=(8, 6))
    for hu, ax in zip(HUs, axs.flatten()):
        plot_delta_auc(res[res.insert_HU == hu], f=f, ax=ax)
        ax.get_legend().remove()
        ax.set_title(f'{hu} HU insert')

    handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5,1.2), ncol=2)
    f.savefig(results_dir/'diffauc_v_diameter_hu.png', dpi=600, facecolor='white', bbox_inches='tight')

# %%
if __name__ == '__main__':

    parser = ArgumentParser(description='Plots LCD results')
    parser.add_argument('results_csv', nargs='?', default=None, type=str, help='csv with lcd auc results, typically output from `measure_LCD_diameter_dependence.m`')
    parser.add_argument('-o','--output_dir', required=False, type=str, help='optional directory to save plots in')
    args = parser.parse_args()

    results_csv = args.results_csv or 'results/test/lcd_v_diameter.csv'
    results_dir = args.output_dir

    main(results_csv, results_dir)