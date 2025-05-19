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

dat = pd.read_csv('/Users/tylerbagwell/Desktop/SpatialAgg_MayDecAnnual_tp_DMItype2_Global_square4_19502023.csv')
stddev = np.std(dat['cindex_lag0y'])
print("... cindex std: ", np.round(stddev,3))

varname = 'tp_anom'
dvarname = str('d'+varname)

df = dat.copy()
df = df.dropna(subset=[dvarname])

# subset to only onset-years
mask = df['conflict_count'] > 0.0
df = df.loc[mask]
replicated_idx = df.index.repeat(df['conflict_count']) # expand onset-years with more than one onset
df = df.loc[replicated_idx].reset_index(drop=True)


# subset to only onset-years
mask = df['conflict_count'] > 0.0
df = df.loc[mask]
replicated_idx = df.index.repeat(df['conflict_count']) # expand onset-years with more than one onset
df = df.loc[replicated_idx].reset_index(drop=True)

print(df.shape)

# plt.scatter(df['tp_anom'], df['tp_diff_std'])
# plt.show()

mask0 = (df['psi'] <= psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
group0 = df.loc[mask0]
n0 = group0.shape[0]

mask1 = (df['psi_tp_directional'] > 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
group1 = df.loc[mask1]
n1 = group1.shape[0]

mask2 = (df['psi_tp_directional'] < 0.0) & (df['psi'] > psi_threshold) & (df['cindex_lag0y'] < -1.0 * stddev)
group2 = df.loc[mask2]
n2 = group2.shape[0]

#### PLOTTING
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

xb = np.median(df[varname]); yb = np.median(df[dvarname])
x0 = np.median(group0[varname]); y0 = np.median(group0[dvarname])
x1 = np.median(group1[varname]); y1 = np.median(group1[dvarname])
x2 = np.median(group2[varname]); y2 = np.median(group2[dvarname])

plt.figure(figsize=(4,3.75))

plt.scatter(group0[varname], group0[dvarname], 
            alpha=0.15, color='gray', edgecolor='gray', s=20, zorder=1, label=f'weakly-affected (n={n0})')
plt.scatter(group1[varname], group1[dvarname], 
            alpha=0.5, color=colors[7], edgecolor=colors[7], s=20, zorder=3, label=f'teleconnected & corr>0 (n={n1})')
plt.scatter(group2[varname], group2[dvarname], 
            alpha=0.5, color=colors[1], edgecolor=colors[1], s=20, zorder=3, label=f'teleconnected & corr<0 (n={n2})')

plt.axvline(0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5, zorder=0)

plt.scatter(x0, y0, color='gray',    s=100, marker='P', edgecolor='white', zorder=4, label='coordinates of 2D median')
plt.scatter(x1, y1, color=colors[7], s=100, marker='P', edgecolor='white', zorder=4)
plt.scatter(x2, y2, color=colors[1], s=100, marker='P', edgecolor='white', zorder=4)

plt.legend(fontsize=7)

plt.xlabel('Precipitation anomaly in s.d.')
plt.ylabel('YoY change in precipitation in s.d.')
plt.title('Onset-years during ENSO NEUTRAL,\nglobal grid cells, 1951-2023')


plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/ENSOneutral_prec_vs_diff_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


# z0 = group0[varname].reset_index(drop=True)
# z1 = group1[varname].reset_index(drop=True)
# z2 = group2[varname].reset_index(drop=True)

# df = pd.DataFrame({
#     'group0': z0,
#     'group1': z1,
#     'group2': z2
# })
# print(df.head())
# df.to_csv("/Users/tylerbagwell/Desktop/groups.txt", sep="\t", index=False, header=True)




