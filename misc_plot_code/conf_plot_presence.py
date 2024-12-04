import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

print('\n\nSTART ---------------------\n')

#######################################
#######################################

import seaborn as sns
cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]
print(colors)

val_min = 0
val_max = 0

path1 = '/Users/tylerbagwell/Desktop/results_Binary_Africa_DMI_hex1d5_CON1_nocontrols_lag0_low_subset.csv'
path2 = '/Users/tylerbagwell/Desktop/results_Binary_Africa_DMI_hex1d5_CON1_nocontrols_lag0_high_subset.csv'

df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

index_closest1 = df1['V1'].abs().idxmin()
index_closest2 = df2['V1'].abs().idxmin()
val1 = 0#df1['estimate__'].iloc[index_closest1]
val2 = 0#df2['estimate__'].iloc[index_closest2]


df1['V2'] = df1['V2'] - 1
df1['97.5%'] = df1['97.5%'] - 1
df1['2.5%'] = df1['2.5%'] - 1

df2['V2'] = df2['V2'] - 1
df2['97.5%'] = df2['97.5%'] - 1
df2['2.5%'] = df2['2.5%'] - 1

# maroon, navy
plt.figure(figsize=(5, 4))
plt.plot(df1['V1'], df1['V2'], color='green', label=r'Weak $(\Psi=0.5)$', linewidth=2.0)
plt.fill_between(df1['V1'], df1['2.5%'], df1['97.5%'], color='green', alpha=0.22, edgecolor=None)

plt.plot(df2['V1'], df2['V2'], color='r', label=r'Strong $(\Psi=1.2)$', linewidth=2.0)
plt.fill_between(df2['V1'], df2['2.5%'], df2['97.5%'], color='r', alpha=0.22, edgecolor=None)


plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, zorder=0)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, zorder=0)

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title('Subset Analysis:\nState-based conflict presence in Africa (1989-2023)\nIOD Lag 0 effect, logit model')
plt.xlabel('Dipole Mode Index (s.d.)', fontsize=12)
plt.ylabel(r'P.P. $\Delta$odds of conflict', fontsize=12)
plt.legend(loc=2, frameon=False, title='Teleconnection strength')

plt.axvspan(+1.5, +2.5, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(+0.5, +1.5, color=colors[3], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-1.5, -0.5, color=colors[1], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
plt.axvspan(-2.5, -1.5, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)

plt.text(0.0, -0.62, 'Neutral', fontsize=10, color='k', horizontalalignment='center')
plt.text(-1., -0.62, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
plt.text(+1., -0.62, 'Moderate', fontsize=10, color='k', horizontalalignment='center')
plt.text(+2., -0.62, 'Strong', fontsize=10, color='k', horizontalalignment='center')
plt.text(-2., -0.62, 'Strong', fontsize=10, color='k', horizontalalignment='center')

plt.xlim(-2.5, +2.5)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/panel_datasets/presence_results/Presence_DMI_Africa_lag0_subset.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()