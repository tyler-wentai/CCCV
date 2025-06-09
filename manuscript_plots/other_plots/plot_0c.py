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

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyMean_tp_DMItype2_Global_square4_19502023.csv')
dat.rename(columns={'tp_total': 'tp_anom'}, inplace=True)
print(dat)

psi_threshold = 0.4

stddev = np.std(dat['cindex_lag0y'])
print(stddev)

#0
mask0 = (
    (dat['conflict_count'] > 0.0) &
    (dat['psi']             <= psi_threshold)
)
anom0 = dat.loc[mask0]

mask0 = anom0['psi_tp_directional'] > 0.0
anom0_p = anom0.loc[mask0]
mask0 = ((anom0_p['cindex_lag0y'] > +1.0 * stddev))
anom0_p = anom0_p.loc[mask0]

mask0 = anom0['psi_tp_directional'] < 0.0
anom0_m = anom0.loc[mask0]
mask0 = ((anom0_m['cindex_lag0y'] < -1.0 * stddev))
anom0_m = anom0_m.loc[mask0]

anom0 = pd.concat([anom0_m, anom0_p], ignore_index=True)



dat = dat[dat['psi'] > psi_threshold]

#1
mask1 = dat['psi_tp_directional'] > 0.0
anom1 = dat.loc[mask1]
mask1 = (
    (anom1['cindex_lag0y'] > +1.0 * stddev)
)
anom1 = anom1.loc[mask1]

mask1 = anom1['conflict_count'] > 0.0
anom1 = anom1.loc[mask1]

# anom1_agg = anom1.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'median',
# }).reset_index()

#2
mask2 = dat['psi_tp_directional'] < 0.0
anom2 = dat.loc[mask2]
mask2 = (
    (anom2['cindex_lag0y'] < -1.0 * stddev)
)
anom2 = anom2.loc[mask2]

mask2 = anom2['conflict_count'] > 0.0
anom2 = anom2.loc[mask2]


# anom2_agg = anom2.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'median',
# }).reset_index()

mean0 = np.median(anom0['tp_anom'])
mean1 = np.median(anom1['tp_anom'])
mean2 = np.median(anom2['tp_anom'])



### --- PLOT 1
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

x0 = anom0["tp_anom"]
x1 = anom1["tp_anom"]
x2 = anom2["tp_anom"]

combined = np.concatenate([x0, x1, x2])
bin_edges = np.histogram_bin_edges(combined, bins="scott")

# 2) plot both with the same bin_edges
plt.figure(figsize=(5,4))
plt.hist(x0,
         bins=bin_edges,
         alpha=0.2,
         label="weakly-affected",
         color='gray',
         #density=True,
         edgecolor='gray')
plt.hist(x1,
         bins=bin_edges,
         alpha=0.4,
         label="teleconnected & corr>0",
         color=colors[7],
         #density=True,
         edgecolor=colors[7])
plt.hist(x2,
         bins=bin_edges,
         alpha=0.4,
         label="teleconnected & corr<0",
         color=colors[1],
         #density=True,
         edgecolor=colors[1])
plt.axvline(mean0, color='gray',    linestyle='--', linewidth=1.5, label=f'{mean0:.3f}')
plt.axvline(mean1, color=colors[7], linestyle='--', linewidth=1.5, label=f'{mean1:.3f}')
plt.axvline(mean2, color=colors[1], linestyle=':',  linewidth=1.5, label=f'{mean2:.3f}')
plt.legend(fontsize=7)
plt.xlabel('Precipitation anomaly in s.d.')
plt.ylabel('Count')
plt.title('Onset-year precipitation anomaly\nIOD drying subset')
# plt.xlim(-2, +2)

plt.tight_layout()
plt.show()