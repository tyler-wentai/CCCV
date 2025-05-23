import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import seaborn as sns
import pandas as pd
import sys
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from shapely import wkt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

print('\n\nSTART ---------------------\n')

path_0 = '/Users/tylerbagwell/Desktop/psithreshold_analysis/data/PsiThreshold_Less_cindex_lag0y_Onset_Count_Global_NINO3_square4_ci90.csv'
path_1 = '/Users/tylerbagwell/Desktop/psithreshold_analysis/data/PsiThreshold_Less_cindex_lag0y_Onset_Count_Global_NINO3type2_square4_ci90.csv'

df0 = pd.read_csv(path_0)
df1 = pd.read_csv(path_1)

import matplotlib.pyplot as plt

x0 = df0["psi_quantile"].values
y0 = df0["estimate"].values
lower0 = df0["conf.low"].values
upper0 = df0["conf.high"].values

x1 = df1["psi_quantile"].values
y1 = df1["estimate"].values
lower1 = df1["conf.low"].values
upper1 = df1["conf.high"].values


###
fig, ax = plt.subplots(figsize=(4.25, 4.5))
color1 = "#ee0235"
color2 = "gray"

ax.plot(x0, y0, marker="o", linestyle="-", markersize=3, linewidth=0.75, color=color1, label="Sum of sig. correlations")
ax.plot(x1, y1, marker="^", linestyle="-", markersize=3, linewidth=0.75, color=color2, label="Max of 3-month running mean")
ax.fill_between(
    x0,
    lower0,
    upper0,
    color=color1,
    alpha=0.40
)
ax.fill_between(
    x1,
    lower1,
    upper1,
    color=color2,
    alpha=0.30
)
ax.axhline(0, color='black', linewidth=1.00)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)  # ensures grid is drawn beneath the plot elements

# second x-axis
x00 = df0["psi_threshold"].values
secax = ax.secondary_xaxis(-0.2, functions=(lambda x00: x00, lambda x00: x00))
secax.set_xlabel('Maximum teleconnection strength')

prim_ticks = ax.get_xticks()
sec_labels = np.interp(prim_ticks, x0, x00)
secax.set_xticks(prim_ticks)
secax.set_xticklabels([f"{v:.2f}" for v in sec_labels])

secax.spines['bottom'].set_color(color1) # color
secax.tick_params(axis='x', colors=color1) # color
secax.xaxis.label.set_color(color1) # color

# second x-axis
x11 = df1["psi_threshold"].values
secax = ax.secondary_xaxis(-0.4, functions=(lambda x11: x11, lambda x11: x11))
secax.set_xlabel('Maximum teleconnection strength')

prim_ticks = ax.get_xticks()
sec_labels = np.interp(prim_ticks, x1, x11)
secax.set_xticks(prim_ticks)
secax.set_xticklabels([f"{v:.2f}" for v in sec_labels])

secax.spines['bottom'].set_color(color2) # color
secax.tick_params(axis='x', colors=color2) # color
secax.xaxis.label.set_color(color2) # color

plt.subplots_adjust(bottom=0.33)

ax.legend(loc=1, title="Teleconnection strength method", fontsize=7)

plt.xlabel('Maximum teleconnection strength (quantile)')
plt.ylabel(r'Estimate of $NINO3_t$ Effect')
plt.title('Sensitivity to max. teleconnection strength\n for defining weakly-affected group\n(ENSO, global grid cells)',
          fontsize=10)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/psithreshold_analysis/plots/MaxThreshold_Global_NINO3_square4_ci90.png', 
            dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
