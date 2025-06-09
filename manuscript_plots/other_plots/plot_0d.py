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
###############

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyMean_tp_DMItype2_Global_square4_19502023.csv')
# dat.rename(columns={'tp_mean': 'tp_anom'}, inplace=True)
print(dat)

df = dat.sort_values(['loc_id', 'year'])

# 2. compute year-to-year difference of tp_mean within each loc_id
df['tp_diff'] = df.groupby('loc_id')['tp_mean'].diff()

# 3. standardize those differences within each loc_id
df['tp_diff_std'] = (
    df
    .groupby('loc_id')['tp_diff']
    .transform(lambda x: (x - x.mean()) / x.std())
)

df = df.dropna(subset=['tp_diff'])

print(df.head(10))




psi_threshold = 0.4

stddev = np.std(dat['cindex_lag0y'])
print(stddev)

#0
mask0 = (
    (df['conflict_count'] > 0.0) &
    (df['psi']             <= psi_threshold)
)
anom0 = df.loc[mask0]

mask0 = anom0['psi_tp_directional'] > 0.0
anom0_p = anom0.loc[mask0]
mask0 = ((anom0_p['cindex_lag0y'] < -1.0 * stddev))
anom0_p = anom0_p.loc[mask0]

mask0 = anom0['psi_tp_directional'] < 0.0
anom0_m = anom0.loc[mask0]
mask0 = ((anom0_m['cindex_lag0y'] < -1.0 * stddev))
anom0_m = anom0_m.loc[mask0]

anom0 = pd.concat([anom0_m, anom0_p], ignore_index=True)


###
df = df[df['psi'] > psi_threshold]

#1
mask1 = df['conflict_count'] > 0.0
anom1 = df.loc[mask1]
mask1 = anom1['psi_tp_directional'] > 0.0
anom1 = anom1.loc[mask1]
mask1 = (
    (anom1['cindex_lag0y'] < -1.0 * stddev)
)
anom1 = anom1.loc[mask1]



#2
mask2 = df['conflict_count'] > 0.0
anom2 = df.loc[mask2]
mask2 = anom2['psi_tp_directional'] < 0.0
anom2 = anom2.loc[mask2]
mask2 = (
    (anom2['cindex_lag0y'] < -1.0 * stddev)
)
anom2 = anom2.loc[mask2]


mean0 = np.median(anom0['tp_diff_std'])
mean1 = np.median(anom1['tp_diff_std'])
mean2 = np.median(anom2['tp_diff_std'])



### --- PLOT 1
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

x0 = anom0["tp_diff_std"]
x1 = anom1["tp_diff_std"]
x2 = anom2["tp_diff_std"]

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

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
proxy0 = (Patch(facecolor='gray', edgecolor='gray', alpha=0.2),
          Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5))
proxy1 = (Patch(facecolor=colors[7], edgecolor=colors[7], alpha=0.4),
          Line2D([0], [0], color=colors[7], linestyle='--', linewidth=1.5))
proxy2 = (Patch(facecolor=colors[1], edgecolor=colors[1], alpha=0.4),
          Line2D([0], [0], color=colors[1], linestyle=':',  linewidth=1.5))

plt.legend([proxy0, proxy1, proxy2],
           [f"Weakly-affected (M={mean0:.2f})",
            f"Teleconnected & corr>0 (M={mean1:.2f})",
            f"Teleconnected & corr<0 (M={mean2:.2f})"],
           handler_map={tuple: HandlerTuple(ndivide=None)},
           fontsize=7)

plt.xlabel('Precipitation year-on-year change in s.d.')
plt.ylabel('Count')
plt.title('Onset-year precipitation anomaly\nIOD drying subset')
# plt.xlim(-2, +2)

plt.tight_layout()
plt.show()