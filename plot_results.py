# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

res = pd.read_csv('results_measure.csv')

sns.lineplot(data=res, x=' dose_level', y=' auc', hue='recon', style='observer')

# %%
