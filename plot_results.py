# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_csv('lcd_results.csv')

sns.lineplot(data=res, x='dose_level', y='auc', hue='recon', style='observer')

# %%
HUs = res.insert_HU.unique()
f, axs = plt.subplots(2, 2, sharex=True,sharey=True, tight_layout=True, figsize=(8, 6))
for hu, ax in zip(HUs, axs.flatten()):
    temp_df = res[res.insert_HU == hu]
    sns.lineplot(ax=ax, data=temp_df, x='dose_level', y='auc', hue='recon', style='observer')
    ax.set_title(f'{hu} HU insert')
f.savefig('LCD_results.png', dpi=300)
# %%
