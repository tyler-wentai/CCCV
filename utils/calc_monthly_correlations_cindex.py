import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from calc_annual_index import *

print('\n\nSTART ---------------------\n')



start_year  = 1950
end_year    = 2024 # need to add one more year

cindex1 = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                       start_date=datetime(start_year, 1, 1, 0, 0, 0),
                       end_date=datetime(end_year, 12, 1, 0, 0, 0))
cindex2 = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
                       start_date=datetime(start_year, 1, 1, 0, 0, 0),
                       end_date=datetime(end_year, 12, 1, 0, 0, 0))

cindex1.index = pd.to_datetime(cindex1.index)
cindex1['Year']  = cindex1.index.year
cindex1['Month'] = cindex1.index.month

cindex2.index = pd.to_datetime(cindex2.index)
cindex2['Year']  = cindex2.index.year
cindex2['Month'] = cindex2.index.month

cindex1_monthly = cindex1.pivot(index='Year', columns='Month', values='ANOM')
for m in range(1, 8):                       # months 1‑4
    cindex1_monthly[f'{m:02d}_next'] = cindex1_monthly[m].shift(-1)
cindex1_monthly = cindex1_monthly.iloc[:-1]

cols = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] +
        ['01_next', '02_next', '03_next', '04_next', '05_next', '06_next', '07_next'])
cindex1_monthly = cindex1_monthly[cols]

cindex1_corr = cindex1_monthly.corr()

cindex2_monthly = cindex2.pivot(index='Year', columns='Month', values='ANOM')
cindex2_corr = cindex2_monthly.corr()

print(cindex1_corr)

fig, ax = plt.subplots(figsize=(7, 7))
plt.suptitle("Monthly Correlations (1950-2023)", fontsize=12)

# Heatmap 1: Lower triangle only
sns.heatmap(
    cindex1_corr,
    annot=True,               # Annotate cells with correlation coefficients
    fmt=".2f",                # Round annotations to two decimals
    cmap='PuOr',               # Blue-to-white-to-red colormap
    vmin=-1, vmax=1,          # Set the color scale limits
    center=0,                 # Center the colormap at 0
    linewidths=0.5,           # Set grid line width
    linecolor='white',         # Change grid line color to gray
    square=True,              # Make cells square
    cbar_kws={"shrink": 0.6},  # Adjust color bar size
    annot_kws={"size": 6}, # Font size for the annotations
    ax=ax
)
ax.set_title("NINO3", fontsize=10)

plt.tight_layout()
# plt.savefig("/Users/tylerbagwell/Desktop/cccv_data/pub_plots/cindex_monthly_correlation_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()

sys.exit()


mask1 = np.triu(np.ones_like(cindex1_corr, dtype=bool), k=1)
mask2 = np.triu(np.ones_like(cindex2_corr, dtype=bool), k=1)

month_mapping = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

cindex1_corr.rename(index=month_mapping, columns=month_mapping, inplace=True)
cindex2_corr.rename(index=month_mapping, columns=month_mapping, inplace=True)

#### PLOTTING

# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = ['LiSong Pro']

fig, axes = plt.subplots(ncols=2, figsize=(9, 4.0))
plt.suptitle("Monthly Correlations (1950-2023)", fontsize=12)

# Heatmap 1: Lower triangle only
sns.heatmap(
    cindex1_corr,
    annot=True,               # Annotate cells with correlation coefficients
    fmt=".2f",                # Round annotations to two decimals
    cmap='PuOr',               # Blue-to-white-to-red colormap
    vmin=-1, vmax=1,          # Set the color scale limits
    center=0,                 # Center the colormap at 0
    linewidths=0.5,           # Set grid line width
    linecolor='white',         # Change grid line color to gray
    square=True,              # Make cells square
    cbar_kws={"shrink": 0.6},  # Adjust color bar size
    annot_kws={"size": 6}, # Font size for the annotations
    ax=axes[0]
)
axes[0].set_title("NINO3", fontsize=10)

# Heatmap 2: Full correlation matrix
sns.heatmap(
    cindex2_corr,
    annot=True,
    fmt=".2f",
    cmap='PuOr',
    vmin=-1, vmax=1,
    center=0,
    linewidths=0.5,
    linecolor='white',
    square=True,
    cbar_kws={"shrink": 0.6},
    annot_kws={"size": 6}, # Font size for the annotations
    ax=axes[1]
)
axes[1].set_title("DMI", fontsize=10)

for ax in axes:
    ax.tick_params(axis='both', which='both', length=0)

plt.tight_layout()
# plt.savefig("/Users/tylerbagwell/Desktop/cccv_data/pub_plots/cindex_monthly_correlation_heatmaps.png", dpi=300, bbox_inches='tight')
plt.show()