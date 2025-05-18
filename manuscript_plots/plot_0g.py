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

psi_threshold = 0.5

dat = pd.read_csv('/Users/tylerbagwell/Desktop/SpatialAgg_JanDecAnnual_t2m_NINO3type2_Global_square4_19502023.csv')
stddev = np.std(dat['cindex_lag0y'])
print("... cindex std: ", np.round(stddev,3))

varname = 't2m_anom'
dvarname = str('d'+varname)

df = dat.copy()
df = df.dropna(subset=[dvarname])



n1 = np.sum(df['conflict_count'])

mask = df['conflict_count'] == 0.0
df_noconflict = df.loc[mask]

# subset to only onset-years
mask = df['conflict_count'] > 0.0
df = df.loc[mask]
replicated_idx = df.index.repeat(df['conflict_count']) # expand onset-years with more than one onset
df = df.loc[replicated_idx].reset_index(drop=True)

print(df.shape)

#### --- 
group0 = df_noconflict
n0 = df_noconflict.shape[0]
group1 = df


#### PLOTTING
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

xb = np.median(df[varname]); yb = np.median(df[dvarname])
x0 = np.median(group0[varname]); y0 = np.median(group0[dvarname])
x1 = np.median(group1[varname]); y1 = np.median(group1[dvarname])

plt.figure(figsize=(4.5,4))

plt.scatter(group0[varname], group0[dvarname], marker='.',
            alpha=0.10, color='gray', edgecolor='gray', s=5, zorder=1, label=f'non-onset-year (n={n0})')
plt.scatter(group1[varname], group1[dvarname], 
            alpha=0.5, color='red', edgecolor='red', s=5, marker='.', zorder=3, label=f'onset-year (n={n1})')

plt.axvline(0, color='k', linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5, zorder=0)

plt.scatter(x0, y0, color='black',    s=50, marker='P', edgecolor='white', zorder=4, label='coordinates of 2D median')
plt.scatter(x1, y1, color='darkred',  s=50, marker='P', edgecolor='white', zorder=4)

print(np.corrcoef((group0[varname], group0[dvarname])))
print(np.corrcoef((group1[varname], group1[dvarname])))

plt.legend(fontsize=7)

plt.xlabel('temperature anomaly in s.d.')
plt.ylabel('YoY change in temperature in s.d.')
plt.title('Global conflict\nYoY change in temp. vs. Annual temp. anomaly', fontsize=11)


plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/Globalconflict_temp_vs_diff_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()






