# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_theme()

res = pd.read_csv('lcd_v_diameter_results.csv')
res = res[~res.duplicated()]
#%%
res.rename(columns={'dose_level': 'dose level [%]'}, inplace=True)
res = res[res.diameter != 200]
f, axs = plt.subplots(1,2)
data_df = res[(res['dose level [%]']==100) | (res['dose level [%]'] ==25)]
sns.lineplot(ax=axs[0], data=data_df[res['observer'] != 'NPW 2D'], x='diameter', y='auc', hue='recon', style='dose level [%]')
sns.lineplot(ax=axs[1], data=data_df[res['observer'] == 'NPW 2D'], x='diameter', y='auc', hue='recon', style='dose level [%]')
f.tight_layout()
f.savefig('auc_v_diameter.png', dpi=600, facecolor='white')
# %%
f, ax = plt.subplots(figsize=(4.5,3.5))
sns.lineplot(ax=ax, data=res[(res.diameter==151)], x='dose level [%]', y='auc', hue='recon', style='diameter') # | (res.diameter == 292) | (res.diameter == 350)
f.tight_layout()
f.savefig('auc_v_dose.png', dpi=600, facecolor='white')
# %%
# res = pd.read_csv('lcd_v_diameter_results.csv')
# res = res[res.diameter != 200]
# res = res[res.diameter.isin(res[res.recon != 'fbp'].diameter.unique())]
# res[res.recon != 'fbp'].auc.to_numpy() - res[res.recon == 'fbp'].auc.to_numpy()
# %%
import numpy as np
f, axs = plt.subplots(1, 2, figsize = (8,4))
sns.lineplot(ax=axs[0], data=res[(res.diameter==151)], x='dose level [%]', y='auc', hue='recon', style='diameter') # | (res.diameter == 292) | (res.diameter == 350)
temp = res[(res.diameter==151)]

# dose_lvl = temp[temp.recon == 'cnn-mse']['dose level [%]'].to_numpy()
ref_auc = temp[temp.recon == 'fbp'].pop('auc')
ref_auc = np.concatenate([ref_auc, ref_auc])
delta_df = temp[temp.recon != 'fbp']
delta_df['$\Delta auc$'] = delta_df['auc'] - ref_auc
delta_df.pop('auc')
delta_df.pop('snr')
sns.lineplot(ax=axs[1], data=delta_df, x='dose level [%]', y = '$\Delta auc$', hue='recon', style='observer')
axs[1].hlines(y=0, xmin=0, xmax=100, color='black', linestyle='--')
axs[1].set_title('Task advantage following denoising')
f.tight_layout()
f.savefig('auc_v_dose_with_comp.png', dpi=600, facecolor='white')
# %%
f, axs = plt.subplots(1, 2, figsize = (8,4))
sns.lineplot(ax=axs[0], data=res[(res['dose level [%]']==25)], x='diameter', y='auc', hue='recon', style='observer') # | (res.diameter == 292) | (res.diameter == 350)
temp = res[(res.diameter != 200)]

# dose_lvl = temp[temp.recon == 'cnn-mse']['dose level [%]'].to_numpy()
ref_auc = temp[temp.recon == 'fbp'].pop('auc')
ref_auc = np.concatenate([ref_auc, ref_auc])
delta_df = temp[temp.recon != 'fbp']
delta_df['$\Delta auc$'] = delta_df['auc'] - ref_auc
delta_df.pop('auc')
delta_df.pop('snr')
sns.lineplot(ax=axs[1], data=delta_df, x='diameter', y = '$\Delta auc$', hue='recon', style='observer')
axs[1].hlines(y=0, xmin=100, xmax=350, color='black', linestyle='--')
axs[1].set_title('Task advantage following denoising')
f.tight_layout()
f.savefig('auc_v_diameter_w_comp.png', dpi=600, facecolor='white')
axs[0].legend(loc='upper center', bbox_to_anchor=(1.05, 1.45),
          ncol=2, fancybox=True, shadow=False)
axs[1].get_legend().remove()
f.savefig('auc_v_diameter_with_comp.png', dpi=600, facecolor='white', bbox_inches='tight')
# %%
HUs = res.insert_HU.unique()
f, axs = plt.subplots(2, 2, sharex=True,sharey=True, tight_layout=True, figsize=(8, 6))
for hu, ax in zip(HUs, axs.flatten()):
    delta_df = res[res.insert_HU == hu]
    delta_df=delta_df[(delta_df.diameter==151) | (delta_df.diameter == 292)]
    sns.lineplot(ax=ax, data=delta_df, x='dose level [%]', y='auc', hue='recon', style='diameter')
    ax.set_title(f'{hu} HU insert')
f.savefig('LCD_results.png', dpi=300)
# %%
HUs = res.insert_HU.unique()
f, axs = plt.subplots(2, 2, sharex=True,sharey=True, tight_layout=True, figsize=(8, 6))
for hu, ax in zip(HUs, axs.flatten()):
    delta_df = res[res.insert_HU == hu]
    delta_df=delta_df[(delta_df['dose level [%]']==100) | (delta_df['dose level [%]']==25)]
    sns.lineplot(ax=ax, data=delta_df, x='diameter', y='auc', hue='recon', style='dose level [%]')
    ax.set_title(f'{hu} HU insert')
# %%
