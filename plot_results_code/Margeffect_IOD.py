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


# ##
# path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets/Onset_Binary_Global_NINO3_square4_wGeometry.csv'
# df = pd.read_csv(path)

# df['geometry'] = df['geometry'].apply(wkt.loads)

# gdf = gpd.GeoDataFrame(df, geometry='geometry')
# gdf.set_crs(epsg=4326, inplace=True)
# gdf_agg =gdf.groupby('loc_id').agg({
#     'geometry': 'first',
#     'psi': 'first',
#     'conflict_binary':'sum',
# }).reset_index()
# gdf_agg = gpd.GeoDataFrame(gdf_agg, geometry='geometry')
# gdf_agg.set_crs(gdf.crs, inplace=True)

# psi_quants = gdf_agg['psi'].quantile([0.0,0.2,0.4,0.6,0.8,1.0])
# print(psi_quants)
# psi_quants = psi_quants.round(3).tolist()
# psi_quants = [0.000, 0.370, 0.965, 1.280, 2.050, 5.037]

# ##


path_ci = "/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_DMI_square4.csv"
path1_h = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_strong_ci90_linear.csv'
path2_h = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_strong_ci90_linear.csv'

path1_l = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_weak_ci90_linear.csv'
path2_l = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_weak_ci90_linear.csv'

df_ci = pd.read_csv(path_ci)

df1_h = pd.read_csv(path1_h)
df2_h = pd.read_csv(path2_h)

df1_l = pd.read_csv(path1_l)
df2_l = pd.read_csv(path2_l)

index_closest1_h = df1_h['cindex_lag0y'].abs().idxmin()
index_closest2_h = df2_h['cindex_lag0y'].abs().idxmin()
val1_h = df1_h['estimate__'].iloc[index_closest1_h]
val2_h = df2_h['estimate__'].iloc[index_closest2_h]

df1_h['estimate__']     = 100*(df1_h['estimate__'])
df1_h['upper__']        = 100*(df1_h['upper__'])
df1_h['lower__']        = 100*(df1_h['lower__'])

df2_h['estimate__']     = 100*(df2_h['estimate__'])
df2_h['upper__']        = 100*(df2_h['upper__'])
df2_h['lower__']        = 100*(df2_h['lower__'])

index_closest1_l = df1_l['cindex_lag0y'].abs().idxmin()
index_closest2_l = df2_l['cindex_lag0y'].abs().idxmin()
val1_l = df1_l['estimate__'].iloc[index_closest1_l]
val2_l = df2_l['estimate__'].iloc[index_closest2_l]

df1_l['estimate__']     = 100*(df1_l['estimate__'])
df1_l['upper__']        = 100*(df1_l['upper__'])
df1_l['lower__']        = 100*(df1_l['lower__'])

df2_l['estimate__']     = 100*(df2_l['estimate__'])
df2_l['upper__']        = 100*(df2_l['upper__'])
df2_l['lower__']        = 100*(df2_l['lower__'])

cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]



# import matplotlib.pyplot as plt
# import seaborn as sns

# # Create a figure with two subplots side by side
# fig = plt.figure(figsize=(9, 6))
# gs = gridspec.GridSpec(2, 2, height_ratios=[1.5, 1])  # Adjust height_ratios as needed

# # ---- Plot for the TOP subplot ----
# ax_top = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())


# gl = ax_top.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
# gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
# gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
# gl.xlabel_style = {'size': 8}
# gl.ylabel_style = {'size': 8}
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# # create a custom colormap
# bounds = psi_quants
# cmap = mcolors.ListedColormap(["#648FFF", "#785EF0", "#FFB000", "#FE6100", "#DC267F"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# index_box = mpatches.Rectangle((-150, -5), 60, 10, 
#                         fill=None, edgecolor='darkviolet', linewidth=1.5,
#                         transform=ccrs.PlateCarree())

# gl.top_labels       = False 
# ax_top.set_global()
# gdf_plot = gdf_agg.plot(
#     column='psi',    
#     cmap=cmap, #'tab20c_r',
#     norm=norm,   
#     legend=True,                   
#     legend_kwds={
#         'label': "Equi-onset-count teleconnection\nstrength quintile", 
#         'orientation': "vertical", 
#         'shrink': 0.6,
#         'ticks': psi_quants
#     },
#     ax=ax_top,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax_top.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax_top.coastlines()
# ax_top.add_patch(index_box)
# cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])



# ###

# ax_left = fig.add_subplot(gs[1, 0])
# ax_left_twin = ax_left.twinx()  # Create twin axis if needed

# ax_right = fig.add_subplot(gs[1, 1])
# ax_right_twin = ax_right.twinx()  # Create twin axis if needed

# # ---- Plot for the BOTTOM LEFT subplot ----
# ax_left = fig.add_subplot(gs[1, 0])
# ax_left_twin = ax_left.twinx()  # Create twin axis if needed

# sns.histplot(x=df_ci['x'], color='silver', ax=ax_left, stat='proportion', bins=12, alpha=1.0, zorder=3)
# ax_left.axvline(0, color='dimgray', linestyle='--', linewidth=1)
# ax_left_twin.axhline(0, color='dimgray', linestyle='--', linewidth=1)
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1_l, color='purple', ax=ax_left_twin)
# ax_left_twin.fill_between(df1_l['cindex_lag0y'], df1_l['lower__'], df1_l['upper__'],
#                            color='purple', alpha=0.30, edgecolor=None)
# ax_left_twin.fill_between(df2_l['cindex_lag0y'], df2_l['lower__'], df2_l['upper__'],
#                            color='purple', alpha=0.30, edgecolor=None)

# ax_left.yaxis.tick_right()                # Move histogram ticks to the right
# ax_left.yaxis.set_label_position("right") # Move histogram label to the right
# ax_left.set_yticks([0, 0.15, 0.30])
# ax_left.set_yticklabels([0, 0.15, 0.30], fontsize=8)
# ax_left.set_ylabel('')
# ax_left.set_xlabel(r"NINO3 May-Dec. ($^{\degree}C$)", fontsize=10, color='black')
# ax_left.tick_params(axis='y', direction='in')

# ax_left_twin.yaxis.tick_left()                 # Move line plot ticks to the left
# ax_left_twin.yaxis.set_label_position("left")  # Move line plot label to the left
# ax_left_twin.set_yticks([-50, 0.0, 50, 100])
# ax_left_twin.set_yticklabels(["-50%", "0%", "+50%", "+100%"], fontsize=10)
# ax_left_twin.tick_params(axis='y', direction='in')
# ax_left_twin.set_ylabel(r"Pct. $\Delta$ACR from neutral phase (%)", fontsize=10, color='black')

# ax_left.axvspan(+0.5, +2.8, color=colors[3], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# ax_left.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
# ax_left.axvspan(-2.05, -0.5, color=colors[1], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)

# # plt.text(+0.0, +110, 'Neutral', fontsize=9, color='k', horizontalalignment='center')
# # plt.text(+1.5,  +110, 'El Niño', fontsize=9, color='k', horizontalalignment='center')
# # plt.text(-1.5,  +110, 'La Niña', fontsize=9, color='k', horizontalalignment='center')

# ax_left.set_ylim(0, 2.5)
# ax_left_twin.set_xlim(-2.05, 2.75)
# ax_left_twin.set_ylim(-100, +125)


# # ---- Plot for the BOTTOM RIGHT subplot ----
# ax_right = fig.add_subplot(gs[1, 1])
# ax_right_twin = ax_right.twinx()  # Create twin axis if needed

# sns.histplot(x=df_ci['x'], color='silver', ax=ax_right, stat='proportion', bins=12, alpha=1.0, zorder=3)
# ax_right.axvline(0, color='dimgray', linestyle='--', linewidth=1)
# ax_right_twin.axhline(0, color='dimgray', linestyle='--', linewidth=1)
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_h, color='purple', ax=ax_right_twin)
# ax_right_twin.fill_between(df1_h['cindex_lag0y'], df1_h['lower__'], df1_h['upper__'],
#                             color='purple', alpha=0.30, edgecolor=None)
# ax_right_twin.fill_between(df2_h['cindex_lag0y'], df2_h['lower__'], df2_h['upper__'],
#                             color='purple', alpha=0.30, edgecolor=None)

# ax_right.yaxis.tick_right()                # Move histogram ticks to the right
# ax_right.yaxis.set_label_position("right") # Move histogram label to the right
# ax_right.set_yticks([0, 0.15, 0.30])
# ax_right.set_yticklabels([0, 0.15, 0.30], fontsize=8)
# ax_right.set_ylabel("Obs. proportion               ", ha='right', fontsize=8, color='black')
# ax_right.set_xlabel(r"NINO3 May-Dec. ($^{\degree}C$)", fontsize=10, color='black')
# ax_right.tick_params(axis='y', direction='in')

# ax_right_twin.yaxis.tick_left()                 # Move line plot ticks to the left
# ax_right_twin.yaxis.set_label_position("left")  # Move line plot label to the left
# ax_right_twin.set_yticks([-50, 0.0, 50, 100])
# ax_right_twin.set_yticklabels(["-50%", "0%", "+50%", "+100%"], fontsize=10)
# ax_right_twin.tick_params(axis='y', direction='in')
# ax_right_twin.set_ylabel(r"")

# ax_right.axvspan(+0.5, +2.8, color=colors[3], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# ax_right.axvspan(-0.5, +0.5, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
# ax_right.axvspan(-2.05, -0.5, color=colors[1], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)

# ax_right.set_ylim(0, 2.5)
# ax_right_twin.set_xlim(-2.05, 2.75)
# ax_right_twin.set_ylim(-100, +125)

# plt.tight_layout()
# plt.show()

# sys.exit()




fig = plt.figure(figsize=(4.5, 3.5))
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()

ax1.set_title('Marginal Effect of the Indian Ocean Dipole (N=73)\nCallahan&Mankin Tele.Calc.', fontsize=10, color='black')

# 
sns.histplot(x=df_ci['cindex_lag0y'], color='gainsboro', ax=ax1, stat='proportion', bins=12, alpha=1.0, zorder=3)
# ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
# ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_l, color='dimgray', ax=ax2)
ax2.fill_between(df2_l['cindex_lag0y'], df2_l['lower__'], df2_l['upper__'], color='dimgray', alpha=0.35, edgecolor=None)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_h, color='red', ax=ax2)
ax2.fill_between(df2_h['cindex_lag0y'], df2_h['lower__'], df2_h['upper__'], color='red', alpha=0.25, edgecolor=None)

# ax2.axvline(0, color='black', linestyle='--', linewidth=1)

# Swap tick positions:
ax1.yaxis.tick_right()                # Move histogram ticks to the right
ax1.yaxis.set_label_position("right") # Move histogram label to the right
ax1.set_yticks([0, 0.15, 0.30])
ax1.set_yticklabels([0, 0.15, 0.30], fontsize=8)
ax1.set_ylabel("Obs. proportion               ", ha='right', fontsize=8, color='black')
ax1.set_xlabel(r"Annualized DMI ($^{\degree}C$)", fontsize=10, color='black')
ax1.tick_params(axis='y', direction='in')

ax2.yaxis.tick_left()                 # Move line plot ticks to the left
ax2.yaxis.set_label_position("left")  # Move line plot label to the left
# ax2.set_yticks([+0.5, +1.0, +1.5, +2.0, +2.5, +3.0, +3.5])
# ax2.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x", "2.5x", "3.0x", "3.5x"], fontsize=10)
ax2.tick_params(axis='y', direction='in')
# ax2.set_ylabel(r"Pct. $\Delta$ACR from neutral phase (%)", fontsize=10, color='black')
ax2.set_ylabel(r'ACR per °C', fontsize=10, color='black')

#
# ax1.axvspan(+1.5, +2.8, color=colors[4], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(+0.4, +2.8, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(-0.4, +0.4, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
ax1.axvspan(-2.05, -0.4, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# ax1.axvspan(-2.05, -1.5, color=colors[0], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)

#
# plt.text(+0.0, +3.5, 'Neutral', fontsize=9, color='k', horizontalalignment='center')
# plt.text(+0.5,  +3.5, 'Pos. Phase', fontsize=9, color='k', horizontalalignment='left')
# plt.text(-0.5,  +3.5, 'Neg. Phase', fontsize=9, color='k', horizontalalignment='right')

line_weak = mlines.Line2D([], [], color='dimgray')
patch_weak = mpatches.Patch(color='dimgray', alpha=0.35)
# Strong: darkorange line and fill
line_strong = mlines.Line2D([], [], color='red')
patch_strong = mpatches.Patch(color='red', alpha=0.25)

# Combine each line and its fill into a tuple
handles = [(line_strong, patch_strong),(line_weak, patch_weak)]
labels = ['Strong', 'Weak']

# Create a combined legend using HandlerTuple to combine the tuple handles
# ax2.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc=[0.05,0.6], fontsize=9, frameon=False, title=r'Teleconnection strength, $\Psi$', title_fontsize=8)

ax1.set_ylim(0, 2.5)
ax2.set_xlim(-1.00, 1.00)
# ax2.set_ylim(0.1, 3.85)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/cindex_margeffect_Onset_Binary_Global_DMI_state_90ci_linearACR_TYPE2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





#######################
#######################

#ALL COUNTRIES





# path_ci = "/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_DMI_square4.csv"
# path1_h = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_DMI_state_group_ALL95.csv'
# path2_h = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_DMI_state_group_ALL95.csv'

# path1_l = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_DMI_state_group_ALL95.csv'
# path2_l = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Binary_Global_DMI_state_group_ALL95.csv'

# df_ci = pd.read_csv(path_ci)

# df1_h = pd.read_csv(path1_h)
# df2_h = pd.read_csv(path2_h)

# df1_l = pd.read_csv(path1_l)
# df2_l = pd.read_csv(path2_l)

# index_closest1_h = df1_h['cindex_lag0y'].abs().idxmin()
# index_closest2_h = df2_h['cindex_lag0y'].abs().idxmin()
# val1_h = df1_h['estimate__'].iloc[index_closest1_h]
# val2_h = df2_h['estimate__'].iloc[index_closest2_h]

# df1_h['estimate__']     = 100*(df1_h['estimate__'])
# df1_h['upper__']        = 100*(df1_h['upper__'])
# df1_h['lower__']        = 100*(df1_h['lower__'])

# df2_h['estimate__']     = 100*(df2_h['estimate__'])
# df2_h['upper__']        = 100*(df2_h['upper__'])
# df2_h['lower__']        = 100*(df2_h['lower__'])

# index_closest1_l = df1_l['cindex_lag0y'].abs().idxmin()
# index_closest2_l = df2_l['cindex_lag0y'].abs().idxmin()
# val1_l = df1_l['estimate__'].iloc[index_closest1_l]
# val2_l = df2_l['estimate__'].iloc[index_closest2_l]

# df1_l['estimate__']     = 100*(df1_l['estimate__'])
# df1_l['upper__']        = 100*(df1_l['upper__'])
# df1_l['lower__']        = 100*(df1_l['lower__'])

# df2_l['estimate__']     = 100*(df2_l['estimate__'])
# df2_l['upper__']        = 100*(df2_l['upper__'])
# df2_l['lower__']        = 100*(df2_l['lower__'])

# cmap = sns.diverging_palette(220, 20, as_cmap=True)
# num_colors = 5
# levels = np.linspace(0, 1, num_colors)
# colors = [cmap(level) for level in levels]



# fig = plt.figure(figsize=(4.5, 3.5))
# ax1 = fig.add_subplot(111)
# ax2 = ax1.twinx()

# ax1.set_title('Marginal Effect of the Indian Ocean Dipole (N=73)\n (All countries)', fontsize=10, color='black')

# # 
# sns.histplot(x=df_ci['cindex_lag0y'], color='gainsboro', ax=ax1, stat='proportion', bins=12, alpha=1.0, zorder=3)
# # ax2.axhline(1, color='gray', linestyle='--', linewidth=1)
# # ax1.axvline(0, color='gray', linestyle='--', linewidth=1)
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_l, color='dimgray', ax=ax2)
# ax2.fill_between(df2_l['cindex_lag0y'], df2_l['lower__'], df2_l['upper__'], color='dimgray', alpha=0.35, edgecolor=None)
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_h, color='red', ax=ax2)
# ax2.fill_between(df2_h['cindex_lag0y'], df2_h['lower__'], df2_h['upper__'], color='red', alpha=0.25, edgecolor=None)

# # ax2.axvline(0, color='black', linestyle='--', linewidth=1)

# # Swap tick positions:
# ax1.yaxis.tick_right()                # Move histogram ticks to the right
# ax1.yaxis.set_label_position("right") # Move histogram label to the right
# ax1.set_yticks([0, 0.15, 0.30])
# ax1.set_yticklabels([0, 0.15, 0.30], fontsize=8)
# ax1.set_ylabel("Obs. proportion               ", ha='right', fontsize=8, color='black')
# ax1.set_xlabel(r"Annualized DMI ($^{\degree}C$)", fontsize=10, color='black')
# ax1.tick_params(axis='y', direction='in')

# ax2.yaxis.tick_left()                 # Move line plot ticks to the left
# ax2.yaxis.set_label_position("left")  # Move line plot label to the left
# # ax2.set_yticks([+0.5, +1.0, +1.5, +2.0, +2.5, +3.0, +3.5])
# # ax2.set_yticklabels(["0.5x", "1.0x", "1.5x", "2.0x", "2.5x", "3.0x", "3.5x"], fontsize=10)
# ax2.tick_params(axis='y', direction='in')
# # ax2.set_ylabel(r"Pct. $\Delta$ACR from neutral phase (%)", fontsize=10, color='black')
# ax2.set_ylabel(r'ACR per °C', fontsize=10, color='black')

# #
# # ax1.axvspan(+1.5, +2.8, color=colors[4], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)
# ax1.axvspan(+0.4, +2.8, color=colors[4], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# ax1.axvspan(-0.4, +0.4, color=colors[2], alpha=0.00, edgecolor='none', linewidth=0.0, zorder=0)
# ax1.axvspan(-2.05, -0.4, color=colors[0], alpha=0.20, edgecolor='none', linewidth=0.0, zorder=0)
# # ax1.axvspan(-2.05, -1.5, color=colors[0], alpha=0.15, edgecolor='none', linewidth=0.0, zorder=0)

# #
# # plt.text(+0.0, +3.5, 'Neutral', fontsize=9, color='k', horizontalalignment='center')
# # plt.text(+0.5,  +3.5, 'Pos. Phase', fontsize=9, color='k', horizontalalignment='left')
# # plt.text(-0.5,  +3.5, 'Neg. Phase', fontsize=9, color='k', horizontalalignment='right')

# line_weak = mlines.Line2D([], [], color='dimgray')
# patch_weak = mpatches.Patch(color='dimgray', alpha=0.35)
# # Strong: darkorange line and fill
# line_strong = mlines.Line2D([], [], color='red')
# patch_strong = mpatches.Patch(color='red', alpha=0.25)

# # Combine each line and its fill into a tuple
# handles = [(line_strong, patch_strong),(line_weak, patch_weak)]
# labels = ['Strong', 'Weak']

# # Create a combined legend using HandlerTuple to combine the tuple handles
# ax2.legend(handles=handles, labels=labels, handler_map={tuple: HandlerTuple(ndivide=1)}, loc=[0.05,0.5], fontsize=9, frameon=False, title=r'Teleconnection strength, $\Psi$', title_fontsize=8)

# ax1.set_ylim(0, 2.5)
# ax2.set_xlim(-1.00, 1.00)
# # ax2.set_ylim(0.1, 3.85)

# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/cindex_margeffect_Onset_Binary_Global_DMI_state_95ci_ALL_linearACR.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()