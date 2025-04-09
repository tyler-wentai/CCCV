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


path_ci = "/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_NINO3_square4.csv"

path_0 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_group_high90.csv'
path_1 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_group_strong90_1strongelninoremoved.csv'
path_2 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_group_strong90_2strongelninoremoved.csv'
path_3 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_group_strong90_3strongelninoremoved.csv'
path_4 = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_NINO3_square4_group_strong90_4strongelninoremoved.csv'

df_ci = pd.read_csv(path_ci)

df0 = pd.read_csv(path_0)
df0['estimate__']     = 1*(df0['estimate__'])/0.004605692   # 0.004605692 is ACR of all global grid points at neutral phase ENSO
df0['upper__']        = 1*(df0['upper__'])/0.004605692
df0['lower__']        = 1*(df0['lower__'])/0.004605692

df1 = pd.read_csv(path_1)
df1['estimate__']     = 1*(df1['estimate__'])/0.004605692   # 0.004605692 is ACR of all global grid points at neutral phase ENSO
df1['upper__']        = 1*(df1['upper__'])/0.004605692
df1['lower__']        = 1*(df1['lower__'])/0.004605692

df2 = pd.read_csv(path_2)
df2['estimate__']     = 1*(df2['estimate__'])/0.004605692   # 0.004605692 is ACR of all global grid points at neutral phase ENSO
df2['upper__']        = 1*(df2['upper__'])/0.004605692
df2['lower__']        = 1*(df2['lower__'])/0.004605692

df3 = pd.read_csv(path_3)
df3['estimate__']     = 1*(df3['estimate__'])/0.004605692   # 0.004605692 is ACR of all global grid points at neutral phase ENSO
df3['upper__']        = 1*(df3['upper__'])/0.004605692
df3['lower__']        = 1*(df3['lower__'])/0.004605692

df4 = pd.read_csv(path_4)
df4['estimate__']     = 1*(df4['estimate__'])/0.004605692   # 0.004605692 is ACR of all global grid points at neutral phase ENSO
df4['upper__']        = 1*(df4['upper__'])/0.004605692
df4['lower__']        = 1*(df4['lower__'])/0.004605692


cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]





fig = plt.figure(figsize=(4.5, 3.5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.set_title('Removing Strong El Niño Years\n(strongly-teleconnected group)', fontsize=10, color='black')

# 
sns.histplot(x=df_ci['x'], color='gainsboro', ax=ax1, stat='proportion', bins=12, alpha=1.0, zorder=3)
ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
#
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df0, color='#648FFF', ax=ax2)
ax2.fill_between(df0['cindex_lag0y'], df0['lower__'], df0['upper__'], color='#648FFF', alpha=0.35, edgecolor=None)
#
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1, color='#DC267F', ax=ax2)
ax2.fill_between(df1['cindex_lag0y'], df1['lower__'], df1['upper__'], color='#DC267F', alpha=0.25, edgecolor=None)
# #
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2, color='#FFB000', ax=ax2)
ax2.fill_between(df2['cindex_lag0y'], df2['lower__'], df2['upper__'], color='#FFB000', alpha=0.25, edgecolor=None)
# #
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df3, color='orangered', ax=ax2)
# ax2.fill_between(df3['cindex_lag0y'], df3['lower__'], df3['upper__'], color='orangered', alpha=0.25, edgecolor=None)
#
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df4, color='#FFB000', ax=ax2)
# ax2.fill_between(df4['cindex_lag0y'], df4['lower__'], df4['upper__'], color='#FFB000', alpha=0.25, edgecolor=None)

# ax2.axvline(0, color='black', linestyle='--', linewidth=1)

# Swap tick positions:
ax1.yaxis.tick_right()                # Move histogram ticks to the right
ax1.yaxis.set_label_position("right") # Move histogram label to the right
ax1.set_yticks([0, 0.15, 0.30])
ax1.set_yticklabels([0, 0.15, 0.30], fontsize=8)
ax1.set_ylabel("Obs. proportion               ", ha='right', fontsize=8, color='black')
ax1.set_xlabel(r"Annualized NINO3 ($^{\degree}C$)", fontsize=10, color='black')
ax1.tick_params(axis='y', direction='in')

ax2.yaxis.tick_left()                 # Move line plot ticks to the left
ax2.yaxis.set_label_position("left")  # Move line plot label to the left
# ax2.set_yticks([+0.5, +1.0, +1.5, +2.0, +2.5, +3.0, +3.5])
# ax2.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x", "2.5x", "3.0x", "3.5x"], fontsize=10)
ax2.tick_params(axis='y', direction='in')
# ax2.set_ylabel(r"Pct. $\Delta$ACR from neutral phase (%)", fontsize=10, color='black')
ax2.set_ylabel(r'ACR$_{group}$ / ACR$_{global}^{netural}$', fontsize=10, color='black')

#
# ax1.axvspan(+1.5, +2.8, color=colors[4], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(+0.5, +2.8, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(-2.05, -0.5, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# ax1.axvspan(-2.05, -1.5, color=colors[0], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)

#
plt.text(+0.0, +3.5, 'Neutral', fontsize=9, color='k', horizontalalignment='center')
plt.text(+1.0,  +3.5, 'El Niño', fontsize=9, color='k', horizontalalignment='left')
plt.text(-1.0,  +3.5, 'La Niña', fontsize=9, color='k', horizontalalignment='right')


line_0 = mlines.Line2D([], [], color='#648FFF')
patch_0 = mpatches.Patch(color='#648FFF', alpha=0.35)
line_1 = mlines.Line2D([], [], color='#DC267F')
patch_1 = mpatches.Patch(color='#DC267F', alpha=0.25)
line_2 = mlines.Line2D([], [], color='#FFB000')
patch_2 = mpatches.Patch(color='#FFB000', alpha=0.35)

# Combine each line and its fill into a tuple
handles = [(line_0, patch_0),(line_1, patch_1),(line_2, patch_2)]
labels = ['All years (N=73)', '2015 removed', '2015, 1997 removed']

# Create a combined legend using HandlerTuple to combine the tuple handles
ax2.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc=[0.05,0.65], fontsize=9, frameon=False)

ax1.set_ylim(0, 2.5)
ax2.set_xlim(-2.00, 2.75)
# ax2.set_ylim(-0.0075, 0.010)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/cindex_margeffect_removingstrongelninos_Onset_Binary_Global_NINO3_square4_90ci.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
