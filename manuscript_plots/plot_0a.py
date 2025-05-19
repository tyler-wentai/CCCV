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
import seaborn as sns
import matplotlib.ticker as mticker

print('\n\nSTART ---------------------\n')

psi_threshold = 0.5

dat = pd.read_csv('/Users/tylerbagwell/Desktop/SpatialAgg_MayDecAnnual_tp_NINO3type2_Global_square4_19502023.csv')


stddev = np.std(dat['cindex_lag0y'])
print("... cindex std: ", np.round(stddev,3))

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
anom_agg_teleconnected_pos = anom_agg[(anom_agg['psi_x'] > psi_threshold) & (anom_agg['psi_tp_directional_x'] > 0)]
anom_agg_teleconnected_neg = anom_agg[(anom_agg['psi_x'] > psi_threshold) & (anom_agg['psi_tp_directional_x'] < 0)]
anom_agg = anom_agg[anom_agg['psi_x'] <= psi_threshold]


fig = plt.figure(figsize=(4.0, 3.5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax3 = ax1.twiny()


ax1.scatter(anom_agg['tp_anom_posphase'], anom_agg['tp_anom_negphase'], 
            alpha=0.10, color='gray', edgecolor='gray', s=15, zorder=1, label='weakly-affected')
ax1.scatter(anom_agg_teleconnected_pos['tp_anom_posphase'], anom_agg_teleconnected_pos['tp_anom_negphase'], 
            alpha=0.5, color=colors[7], edgecolor=colors[7], s=15, zorder=3, label='teleconnected & corr>0')
ax1.scatter(anom_agg_teleconnected_neg['tp_anom_posphase'], anom_agg_teleconnected_neg['tp_anom_negphase'], 
            alpha=0.5, color=colors[1], edgecolor=colors[1], s=15, zorder=3, label='teleconnected & corr<0')

ax1.axvline(0.0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
ax1.axhline(0.0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
x = np.linspace(-2, 2, 100)
y = -x
ax1.plot(x, y, color='red', linestyle='--', linewidth=1.5, zorder=4)
ax1.set_xlim(-2.3, +2.3)
ax1.set_ylim(-2.3, +2.3)

x0 = np.median(anom_agg['tp_anom_posphase']); y0 = np.median(anom_agg['tp_anom_negphase'])
x1 = np.median(anom_agg_teleconnected_pos['tp_anom_posphase']); y1 = np.median(anom_agg_teleconnected_pos['tp_anom_negphase'])
x2 = np.median(anom_agg_teleconnected_neg['tp_anom_posphase']); y2 = np.median(anom_agg_teleconnected_neg['tp_anom_negphase'])
ax1.scatter(x0, y0, color='gray',  s=100, marker='P', edgecolor='white', zorder=4, linewidth=1.5,
            label=r'Group medians')
ax1.scatter(x1, y1, color=colors[7], s=100, marker='P', edgecolor='white', zorder=4, linewidth=1.5)
ax1.scatter(x2, y2, color=colors[1], s=100, marker='P', edgecolor='white', zorder=4, linewidth=1.5)

ax1.text(1.18, -1.40,
        "line of parity",
        rotation=-39, color='red',
        ha="center", va="center", fontsize=7)

# hist x-axis
all_vals = np.concatenate([anom_agg_teleconnected_pos['tp_anom_negphase'].values, 
                          anom_agg_teleconnected_neg['tp_anom_negphase'].values,
                          anom_agg_teleconnected_pos['tp_anom_posphase'].values, 
                          anom_agg_teleconnected_neg['tp_anom_posphase'].values])
minv, maxv = all_vals.min(), all_vals.max()
bin_edges = np.linspace(minv, maxv, 16)   # 12 equal‐width bins
sns.histplot(x=anom_agg_teleconnected_pos['tp_anom_posphase'], color=colors[7], edgecolor='white',
             ax=ax2, linewidth=0.5, stat='proportion', bins=bin_edges, alpha=0.40, zorder=0)
sns.histplot(x=anom_agg_teleconnected_neg['tp_anom_posphase'], color=colors[1], edgecolor='white',
             ax=ax2, linewidth=0.5, stat='proportion',bins=bin_edges, alpha=0.40, zorder=0)

ax2.set_ylim(0, 2.5)
ax2.set_yticks([])
ax2.set_ylabel(None)

# hist y-axis
sns.histplot(y=anom_agg_teleconnected_pos['tp_anom_negphase'], color=colors[7], edgecolor='white',
             ax=ax3, linewidth=0.5, stat='proportion', bins=bin_edges, alpha=0.40, zorder=0)
sns.histplot(y=anom_agg_teleconnected_neg['tp_anom_negphase'], color=colors[1], edgecolor='white',
             ax=ax3, linewidth=0.5, stat='proportion',bins=bin_edges, alpha=0.40, zorder=0)

ax3.set_xlim(0, 2.5)
ax3.set_xticks([])
ax3.set_xlabel(None)

ax1.set_xlabel(r'Median $P_{anom}$ in El Niño (s.d.)')
ax1.set_ylabel(r'Median $P_{anom}$ in La Niña (s.d.)')
plt.title(r'ENSO, Global grid cells')
ax1.legend(fontsize=6.8, frameon=True)

ax1.set_xticks([-2,-1,0,1,2])
ax1.set_yticks([-2,-1,0,1,2])
ax1.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
ax1.yaxis.set_minor_locator(mticker.MultipleLocator(0.5))

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/NINO3_mediananom_scatter.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
