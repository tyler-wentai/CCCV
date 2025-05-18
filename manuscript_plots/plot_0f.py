import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point
import shapely
from scipy.signal import detrend

print('\n\nSTART ---------------------\n')

###############
### COMPUTES THE STANDARDIZED DIFFERENCE IN PRECIPITATION BETWEEN YEAR t and YEAR t-1
### AND COMPARES WEAKLY-AFFECTED AND TELECONNECTED IN POSITIVE OR NEGATIVE PHASE.
### OUTPUT: SCATTER PLOT
###############

psi_threshold = 0.4

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyT2M_DMItype2_Global_square4_19502023.csv')
stddev = np.std(dat['cindex_lag0y'])
print("... cindex std: ", np.round(stddev,3))

df = dat.sort_values(['loc_id', 'year'])
df['tp_diff'] = df.groupby('loc_id')['tp_mean'].diff()
df['tp_diff_std'] = (
    df
    .groupby('loc_id')['tp_diff']
    .transform(lambda x: (x - x.mean()) / x.std())
)
df = df.dropna(subset=['tp_diff'])

# subset to only onset-years
mask = df['conflict_count'] > 0.0
df = df.loc[mask]
replicated_idx = df.index.repeat(df['conflict_count']) # expand onset-years with more than one onset
df = df.loc[replicated_idx].reset_index(drop=True)

print(df.shape)

#### --- A. WEAKLY AFFECTED ALL
mask0 = (df['psi'] < psi_threshold) & (np.abs(df['cindex_lag0y']) < +1.0 * stddev)
group0 = df.loc[mask0]
n0 = np.sum(group0['conflict_count'])
mask1 = (df['psi'] < psi_threshold) & (df['cindex_lag0y'] > +1.0 * stddev)
group1 = df.loc[mask1]
n1 = np.sum(group1['conflict_count'])
mask2 = (df['psi'] < psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
group2 = df.loc[mask2]
n2 = np.sum(group2['conflict_count'])

#### --- B. TELECONNECTED & CORR>0
# mask0 = (df['psi_tp_directional'] > 0.0) & (df['psi'] > psi_threshold) & (np.abs(df['cindex_lag0y']) < +1.0 * stddev)
# group0 = df.loc[mask0]
# n0 = np.sum(group0['conflict_count'])
# mask1 = (df['psi_tp_directional'] > 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] > +1.0 * stddev)
# group1 = df.loc[mask1]
# n1 = np.sum(group1['conflict_count'])
# mask2 = (df['psi_tp_directional'] > 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
# group2 = df.loc[mask2]
# n2 = np.sum(group2['conflict_count'])

#### --- B. TELECONNECTED & CORR<0
# mask0 = (df['psi_tp_directional'] < 0.0) & (df['psi'] > psi_threshold) & (np.abs(df['cindex_lag0y']) < +1.0 * stddev)
# group0 = df.loc[mask0]
# n0 = np.sum(group0['conflict_count'])
# mask1 = (df['psi_tp_directional'] < 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] > +1.0 * stddev)
# group1 = df.loc[mask1]
# n1 = np.sum(group1['conflict_count'])
# mask2 = (df['psi_tp_directional'] < 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
# group2 = df.loc[mask2]
# n2 = np.sum(group2['conflict_count'])

#### PLOTTING
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

xb = np.median(df['tp_anom']); yb = np.median(df['tp_diff_std'])
x0 = np.median(group0['tp_anom']); y0 = np.median(group0['tp_diff_std'])
x1 = np.median(group1['tp_anom']); y1 = np.median(group1['tp_diff_std'])
x2 = np.median(group2['tp_anom']); y2 = np.median(group2['tp_diff_std'])

plt.figure(figsize=(4.5,4))

plt.scatter(group0['tp_anom'], group0['tp_diff_std'], 
            alpha=0.25, color='gray', edgecolor='gray', s=30, zorder=1, label=f'neutral IOD onset-year (n={n0})')
plt.scatter(group1['tp_anom'], group1['tp_diff_std'], 
            alpha=0.5, color='red', edgecolor='red', s=30, marker='^', zorder=3, label=f'positive IOD onset-year (n={n1})')
plt.scatter(group2['tp_anom'], group2['tp_diff_std'], 
            alpha=0.5, color='blue', edgecolor='blue', s=30, marker='v', zorder=3, label=f'negative IOD onset-year (n={n2})')

plt.axvline(0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5, zorder=0)

plt.scatter(x0, y0, color='black',    s=100, marker='P', edgecolor='white', zorder=4, label='coordinates of 2D median')
plt.scatter(x1, y1, color='darkred',  s=100, marker='P', edgecolor='white', zorder=4)
plt.scatter(x2, y2, color='darkblue', s=100, marker='P', edgecolor='white', zorder=4)

plt.legend(fontsize=7)

plt.xlabel('Precipitation anomaly in s.d.')
plt.ylabel('Year-over-year change in precipitation in s.d.')
plt.title('Onset-years in teleconnected and corr<0\nglobal grid cells')


plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/ENSO_teleconnectedcorrneg_prec_vs_diff_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()






