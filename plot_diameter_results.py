# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_csv('lcd_v_diameter_results.csv')
res = res[res.diameter != 200]
sns.lineplot(data=res[(res.dose_level==100) | (res.dose_level ==25)], x='diameter', y='auc', hue='recon', style='dose_level')
# %%
sns.lineplot(data=res[(res.diameter==151) | (res.diameter == 292) | (res.diameter == 350)], x='dose_level', y='auc', hue='recon', style='diameter')

# %%

HUs = res.insert_HU.unique()
f, axs = plt.subplots(2, 2, sharex=True,sharey=True, tight_layout=True, figsize=(8, 6))
for hu, ax in zip(HUs, axs.flatten()):
    temp_df = res[res.insert_HU == hu]
    sns.lineplot(ax=ax, data=temp_df, x='dose_level', y='auc', hue='recon', style='observer')
    ax.set_title(f'{hu} HU insert')
f.savefig('LCD_results.png', dpi=300)
# %%
