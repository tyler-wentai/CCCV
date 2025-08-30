import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from shapely import wkt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.feature as cfeature
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from calc_annual_index import *

print('\n\nSTART ---------------------\n')



####################################
####################################

path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4_wGeometry.csv'
df = pd.read_csv(path)

df['geometry'] = df['geometry'].apply(wkt.loads)

# Create a GeoDataFrame, specifying the geometry column
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Optionally, set the coordinate reference system (CRS) if known, for example WGS84
gdf.set_crs(epsg=4326, inplace=True)


gdf_agg =gdf.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first',
    'conflict_count':'sum',
}).reset_index()

# Convert the aggregated DataFrame back into a GeoDataFrame and set the active geometry column
gdf_agg = gpd.GeoDataFrame(gdf_agg, geometry='geometry')

# Optionally, set the CRS using the CRS from the original GeoDataFrame
gdf_agg.set_crs(gdf.crs, inplace=True)

onset_path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/conflict_datasets/GeoArmedConflictOnset_v1_CLEANED.csv'
df_onset = pd.read_csv(onset_path)    
gdf_onset = gpd.GeoDataFrame(
    df_onset, 
    geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
    crs="EPSG:4326"
)


start_year, end_year = 1950, 2023

nino3_monthly = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
dmi_monthly = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))

nino3_yearly        = compute_annualized_index('nino3', start_year, end_year+1)
dmi_yearly          = compute_annualized_index('dmi', start_year, end_year)
dmi_noenso_yearly  = compute_annualized_index('dmi_noenso', start_year, end_year)

nino3_std   = nino3_yearly['cindex'].std()
dmi_std     = dmi_yearly['cindex'].std()

sigma_nino3 = nino3_monthly['ANOM'].std()
sigma_dmi = dmi_monthly['ANOM'].std()

nino3_yearly['year']        = pd.to_datetime(nino3_yearly['year'].astype(str), format='%Y')
dmi_yearly['year']          = pd.to_datetime(dmi_yearly['year'].astype(str), format='%Y')
dmi_noenso_yearly['year']   = pd.to_datetime(dmi_noenso_yearly['year'].astype(str), format='%Y')

dmi_yearly['cindex'] = dmi_yearly['cindex'] / dmi_std # STANDARDIZE!!!

import seaborn as sns
cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 3
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

# assign colors
def pick_color_nino3(x):
    if x < -sigma_nino3:
        return 'blue'
    elif x > sigma_nino3:
        return 'red'
    else:
        return 'gray'
def pick_color_dmi(x):
    if x < -1: #-sigma_dmi:
        return colors[0]
    elif x > +1: #sigma_dmi:
        return colors[2]
    else:
        return 'gray'
colors_nino3 = nino3_yearly['cindex'].map(pick_color_nino3)
colors_dmi = dmi_yearly['cindex'].map(pick_color_dmi)


# build legend patches
legend_patches_nino3 = [
    Patch(color='red',   alpha=0.6, label='El Niño'),
    Patch(color='gray',  alpha=0.6, label='Neutral'),
    Patch(color='blue',  alpha=0.6, label='La Niña'),
]
legend_patches_dmi = [
    Patch(color=colors[2],   alpha=0.6, label='+IOD'),
    Patch(color='gray',  alpha=0.6, label='Neutral'),
    Patch(color=colors[0],  alpha=0.6, label='-IOD'),
]


# Define a polygon with lat/lon coordinates
fig = plt.figure(figsize=(8.0, 10))

# ── lay out a 3 × 1 grid ───────────────────────────────
gs   = fig.add_gridspec(nrows=3, ncols=1, hspace=0.3, height_ratios=[3, 1, 1])

ax1  = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())  # map axis
ax2  = fig.add_subplot(gs[1, 0])                              # normal axis
ax3  = fig.add_subplot(gs[2, 0])                              # normal axis

# fig, ax1 = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
gl = ax1.gridlines(
crs=ccrs.PlateCarree(),
draw_labels={
    'bottom': True,
    'left':   True,
    'top':    False,
    'right':  False
},
linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 30))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 61, 30))    # parallels every 30°
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels   = False 
gl.right_labels = False


ax1.set_global()
ax1.coastlines()

ax1.add_feature(cfeature.LAND, facecolor='gainsboro', alpha=0.5)
# ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')

index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                               fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
index_box1 = mpatches.Rectangle((50, -10), 20, 20,
                                fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
index_box2 = mpatches.Rectangle((90, -10), 20, 10, 
                                fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
ax1.add_patch(index_box)
ax1.add_patch(index_box1)
ax1.add_patch(index_box2)





gdf_onset['decade_start'] = (gdf_onset['year'] // 10) * 10
decades = sorted(gdf_onset['decade_start'].unique())
print(decades)

cmap = plt.get_cmap('rainbow', len(decades))
markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h']


x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
max_year = gdf_onset['year'].max()
counts = []
for i, dec in enumerate(decades):
    subset = gdf_onset[gdf_onset['decade_start'] == dec]
    counts.append(subset.shape[0])
    end = min(dec + 9, max_year)              # cap at 2023
    label = f"{dec}-{end}"
    ax1.scatter(
        subset['onset_lon'],
        subset['onset_lat'],
        marker=markers[i % len(markers)],
        s=15,                    # you can tweak size
        color=cmap(i),
        transform=ccrs.PlateCarree(),
        zorder=5,
        label=label,
        alpha=0.75,
        edgecolor='k',
        linewidth=0.5
    )

ax1.set_title('Spatio-Temporal Distribution of Conflict Onsets, 1950-2023', fontsize=12)

# bar plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
axmap = ax1                          # <- the Cartopy axis you already have
axbar = inset_axes(
    axmap,
    width="14%", height="19%",      # 25 % of the map-axis width, full height
    loc="lower left",               # attach on the left of the inset-box …
    bbox_to_anchor=(0.15, 0.20, 1, 1),  # … which we shift just outside (x=1.02)
    bbox_transform=axmap.transAxes,
    borderpad=0,
)

# decades = ['50', '60', '70', '80', '90', '00', '10', '20']
x_pos = np.arange(len(counts))
axbar.bar(x_pos, counts, color="gray", alpha=0.6)
axbar.set_xticks(x_pos)
axbar.set_xticklabels([f"{d}s" for d in decades], rotation=90, fontsize=7)
axbar.tick_params(axis="y", labelrotation=90, labelsize=7)
axbar.set_ylabel("No. of\nonsets", fontsize=8)

axbar.tick_params(axis="both",          # "x", "y", or "both"
               which="both",         # major & minor
               direction="in")
# axbar.spines["left"].set_visible(False)   # aesthetic: hide shared border


#
ax1.text(0.090, 0.485, 'NINO3 region', transform=ax1.transAxes, fontsize=9)
ax1.text(0.655, 0.34, 'DMI regions',  transform=ax1.transAxes, fontsize=9)

# ax.plot([0.66, 0.72], [0.45, 0.39], color='k', linewidth=1.5, transform=ax.transAxes)
from matplotlib.patches import FancyArrowPatch
arrow1 = FancyArrowPatch(
    [0.72, 0.39], [0.66, 0.45], 
    arrowstyle="-|>", mutation_scale=10,
    color="k", lw=1,
    transform=ax1.transAxes   # omit if not on a map
)
arrow2 = FancyArrowPatch(
    [0.72, 0.39], [0.79, 0.45], 
    arrowstyle="-|>", mutation_scale=10,
    color="k", lw=1,
    transform=ax1.transAxes   # omit if not on a map
)
ax1.add_patch(arrow1)
ax1.add_patch(arrow2)


plt.subplots_adjust(right=0.8)  
ax1.legend(
    title="Onset decade",
    title_fontsize=10,
    fontsize=10,
    loc="upper center",           # anchor the *top-centre* of the legend…
    bbox_to_anchor=(0.5, -0.08),  # …to a point 8 % of the axis height below it
    borderaxespad=0,
    framealpha=0.7,
    markerscale=1.75,
    frameon=False,
    ncol=4                        # spread labels in a row if you like
)

ax1.text(+0.00, 1.0, 'a', transform=ax1.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width


# — Panel B —
ax2.plot(
    nino3_yearly['year'],
    nino3_yearly['cindex'],
    color='black',
    linewidth=1.5,
    zorder=1
)
ax2.bar(
    nino3_yearly['year'],
    nino3_yearly['cindex'],
    width=300,
    alpha=0.6,
    color=colors_nino3,
    zorder=0
)
ax2.axhline(0, color='black', linewidth=1.0, linestyle='-')
ax2.set_ylabel('NDJ-Avg. NINO3 (°C)')
ax2.set_title('Annualized Index for the El Niño-Southern Oscillation (ENSO)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.text(+0.00, 1.15, 'b', transform=ax2.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width
ax2.legend(
    handles=legend_patches_nino3,
    loc='upper left',
    frameon=False,
    fontsize=9
)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# — Panel C —
ax3.plot(
    dmi_yearly['year'],
    dmi_yearly['cindex'],
    color='black',
    linewidth=1.5,
    zorder=1
)
ax3.plot(
    dmi_noenso_yearly['year'],
    dmi_noenso_yearly['cindex'],
    color='purple',
    linestyle='--',
    linewidth=1.25,
    label=r'ENSO-Independent DMI',
    zorder=0
)
ax3.bar(
    dmi_yearly['year'],
    dmi_yearly['cindex'],
    width=300,
    alpha=0.6,
    color=colors_dmi,
    zorder=0
)
leg1 = ax3.legend(loc='lower right', frameon=False, fontsize=9)  # plot labels
leg2 = ax3.legend(handles=legend_patches_dmi, loc='upper left', frameon=False, fontsize=9)
ax3.add_artist(leg1)
ax3.set_yticks([-2, -1, 0, 1, 2])
ax3.axhline(0, color='black', linewidth=1.0, linestyle='-')
ax3.set_ylabel('SON-Avg. DMI (s.d.)')
ax3.set_title('Annualized Index for the Indian Ocean Dipole (IOD)')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.text(+0.00, 1.15, 'c', transform=ax3.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width
ax3.set_xlabel('Year')
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/manuscript_plots/Main_fig1_v2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.savefig('/Users/tylerbagwell/Desktop/Main_fig1_v2.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
