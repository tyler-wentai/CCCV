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

dat1 = pd.read_csv('/Users/tylerbagwell/Desktop/SpatialAgg_JanDecAnnual_tp_NINO3type2_Global_square4_19502023.csv')
dat2 = pd.read_csv('/Users/tylerbagwell/Desktop/SpatialAgg_JanDecAnnual_t2m_NINO3type2_Global_square4_19502023.csv')

dat2 = dat2[['index', 't2m_anom', 'dt2m_anom']]
dat = pd.merge(dat1, dat2, on='index', how='inner')

varname1 = 'tp_anom'
varname2 = 't2m_anom'

df = dat.copy()
df = df.dropna(subset=[varname2])



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

xb = np.median(df[varname1]); yb = np.median(df[varname2])
x0 = np.median(group0[varname1]); y0 = np.median(group0[varname2])
x1 = np.median(group1[varname1]); y1 = np.median(group1[varname2])

plt.figure(figsize=(4.5,4))

plt.scatter(group0[varname1], group0[varname2], marker='.',
            alpha=0.10, color='gray', edgecolor='gray', s=5, zorder=1, label=f'non-onset-year (n={n0})')
plt.scatter(group1[varname1], group1[varname2], 
            alpha=0.5, color='red', edgecolor='red', s=5, marker='.', zorder=3, label=f'onset-year (n={n1})')

plt.axvline(0, color='k', linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5, zorder=0)

plt.scatter(x0, y0, color='black',    s=50, marker='P', edgecolor='white', zorder=4, label='coordinates of 2D median')
plt.scatter(x1, y1, color='darkred',  s=50, marker='P', edgecolor='white', zorder=4)

print(np.corrcoef((group0[varname1], group0[varname2])))
print(np.corrcoef((group1[varname1], group1[varname2])))

plt.legend(fontsize=7)

plt.xlabel('precipitation anomaly in s.d.')
plt.ylabel('temperature anomaly in s.d.')
plt.title('Annual temp. anomaly vs. Annual precip. anomaly', fontsize=11)


plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/Globalconflict_temp_vs_precip_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()






