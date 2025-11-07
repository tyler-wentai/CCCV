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

dat = pd.read_csv('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_mrsos_NINO3type2_Global_square4_19502023.csv')
# dat.rename(columns={'tp_total': 'tp_anom'}, inplace=True)
print(dat)

dat = dat.dropna()

threshold = 0.05

stddev = np.std(dat['cindex_lag0y'])
print(stddev)


#1
mask1 = dat['conflict_binary'] > 0.0
anom1 = dat.loc[mask1]
mask1 = (anom1['psi_tp_directional'] < -threshold) & (anom1['cindex_lag0y'] > +1.0)
anom1 = anom1.loc[mask1]
# mask1 = (
#     (anom1['cindex_lag0y'] < -1.0 * stddev)
# )
# anom1 = anom1.loc[mask1]



# anom1_agg = anom1.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'median',
# }).reset_index()

#2
mask2 = dat['conflict_binary'] > 0.0
anom2 = dat.loc[mask2]
mask2 = (anom2['psi_tp_directional'] > +threshold) & (anom2['cindex_lag0y'] < -1.0)
anom2 = anom2.loc[mask2]
# mask1 = (
#     (anom1['cindex_lag0y'] < -1.0 * stddev)
# )
# anom1 = anom1.loc[mask1]


# anom2_agg = anom2.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'median',
# }).reset_index()

mean1 = np.median(anom1['tp_anom'])
mean2 = np.median(anom2['tp_anom'])



### --- PLOT 1
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

x1 = anom1["tp_anom"]
x2 = anom2["tp_anom"]

combined = np.concatenate([x1, x2])
bin_edges = np.histogram_bin_edges(combined, bins=8)


combined = np.concatenate([x1, x2])
lo = np.nanmin(combined)
hi = np.nanmax(combined)
span = max(abs(lo), abs(hi))          # symmetric range around 0
bin_edges = np.linspace(-span, span, 8 + 1)  # 8 bins → 9 edges, 0 included

# 2) plot both with the same bin_edges
plt.figure(figsize=(5,4))
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
plt.axvline(mean1, color=colors[7], linestyle='--', linewidth=1.5, label=f'{mean1:.3f}')
plt.axvline(mean2, color=colors[1], linestyle=':',  linewidth=1.5, label=f'{mean2:.3f}')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
proxy1 = (Patch(facecolor=colors[7], edgecolor=colors[7], alpha=0.4),
          Line2D([0], [0], color=colors[7], linestyle='--', linewidth=1.5))
proxy2 = (Patch(facecolor=colors[1], edgecolor=colors[1], alpha=0.4),
          Line2D([0], [0], color=colors[1], linestyle=':',  linewidth=1.5))

plt.legend([proxy1, proxy2],
           [f"Teleconnected & corr>0 (M={mean1:.2f})",
            f"Teleconnected & corr<0 (M={mean2:.2f})"],
           handler_map={tuple: HandlerTuple(ndivide=None)},
           fontsize=7)


# plt.legend(fontsize=7)
plt.xlabel('← Dryer                   Wetter →\nPrecipitation anomaly in s.d.')
plt.ylabel('Count')
plt.title('Onset-year precipitation anomaly\nENSO all conflicts')
plt.xlim(-3, +3)

# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/NINO3_onsetyear_tpanom_ALL.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.tight_layout()
plt.show()