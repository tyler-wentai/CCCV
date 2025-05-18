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

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv')

stddev = np.std(dat['cindex_lag0y'])
print(stddev)

#1
mask1 = (
    (dat['cindex_lag0y'] > +1.0 * stddev)
)
anom1 = dat.loc[mask1]
anom1_agg = anom1.groupby('loc_id').agg({
    'psi': 'first',
    'psi_tp_directional': 'first',
    'tp_anom':'median',
}).reset_index()

#2
mask2 = (
    (dat['cindex_lag0y'] < -1.0 * stddev)
)
anom2 = dat.loc[mask2]
anom2_agg = anom2.groupby('loc_id').agg({
    'psi': 'first',
    'psi_tp_directional': 'first',
    'tp_anom':'median',
}).reset_index()


#### PLOTTING

cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

anom1_agg.rename(columns={'tp_anom': 'tp_anom_posphase'}, inplace=True)
anom2_agg.rename(columns={'tp_anom': 'tp_anom_negphase'}, inplace=True)

anom_agg = anom1_agg.merge(anom2_agg, on='loc_id')
psi_threshold = 0.4
anom_agg_teleconnected_pos = anom_agg[(anom_agg['psi_x'] > psi_threshold) & (anom_agg['psi_tp_directional_x'] > 0)]
anom_agg_teleconnected_neg = anom_agg[(anom_agg['psi_x'] > psi_threshold) & (anom_agg['psi_tp_directional_x'] < 0)]
anom_agg = anom_agg[anom_agg['psi_x'] <= psi_threshold]



plt.figure(figsize=(4.5,4))

plt.scatter(anom_agg['tp_anom_posphase'], anom_agg['tp_anom_negphase'], 
            alpha=0.15, color='gray', edgecolor='gray', s=20, zorder=1, label='weakly-affected')
plt.scatter(anom_agg_teleconnected_pos['tp_anom_posphase'], anom_agg_teleconnected_pos['tp_anom_negphase'], 
            alpha=0.5, color=colors[7], edgecolor=colors[7], s=20, zorder=3, label='teleconnected & corr>0')
plt.scatter(anom_agg_teleconnected_neg['tp_anom_posphase'], anom_agg_teleconnected_neg['tp_anom_negphase'], 
            alpha=0.5, color=colors[1], edgecolor=colors[1], s=20, zorder=3, label='teleconnected & corr<0')

plt.axvline(0.0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(0.0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
x = np.linspace(-2, 2, 100)
y = -x
plt.plot(x, y, color='red', linestyle='--', linewidth=1.5, zorder=4)
plt.xlim(-2.2, +2.2)
plt.ylim(-2.2, +2.2)

x0 = np.median(anom_agg['tp_anom_posphase']); y0 = np.median(anom_agg['tp_anom_negphase'])
x1 = np.median(anom_agg_teleconnected_pos['tp_anom_posphase']); y1 = np.median(anom_agg_teleconnected_pos['tp_anom_negphase'])
x2 = np.median(anom_agg_teleconnected_neg['tp_anom_posphase']); y2 = np.median(anom_agg_teleconnected_neg['tp_anom_negphase'])
plt.scatter(x0, y0, color='gray',    s=100, marker='P', edgecolor='white', zorder=4, label='coordinates of 2D median')
plt.scatter(x1, y1, color=colors[7], s=100, marker='P', edgecolor='white', zorder=4)
plt.scatter(x2, y2, color=colors[1], s=100, marker='P', edgecolor='white', zorder=4)


plt.xlabel(r'Median $P_{anom}$ in El Niño (s.d.)')
plt.ylabel(r'Median $P_{anom}$ in La Niña (s.d.)')
plt.title(r'ENSO, Global grid cells')
plt.text(1.10, -1.25,
        "line of parity",
        rotation=-41, color='red',
        ha="center", va="center", fontsize=7)
plt.legend(fontsize=7)

plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/NINO3_mediananom_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


# z0 = anom_agg['tp_anom_posphase'].reset_index(drop=True)
# z1 = anom_agg_teleconnected_pos['tp_anom_posphase'].reset_index(drop=True)
# z2 = anom_agg_teleconnected_neg['tp_anom_posphase'].reset_index(drop=True)

# df = pd.DataFrame({
#     'group0': z0,
#     'group1': z1,
#     'group2': z2
# })
# print(df.head())
# df.to_csv("/Users/tylerbagwell/Desktop/groups.txt", sep="\t", index=False, header=True)