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

print('\n\nSTART ---------------------\n')



####################################
####################################

path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4_wGeometry.csv'
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

onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
df_onset = pd.read_csv(onset_path)    
gdf_onset = gpd.GeoDataFrame(
    df_onset, 
    geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
    crs="EPSG:4326"
)


# Define a polygon with lat/lon coordinates
fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
gl = ax.gridlines(
crs=ccrs.PlateCarree(),
draw_labels={
    'bottom': True,
    'left':   True,
    'top':    False,
    'right':  False
},
linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 61, 30))    # parallels every 30°
gl.xlabel_style = {'size': 10}
gl.ylabel_style = {'size': 10}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels   = False 
gl.right_labels = False


ax.set_global()
ax.coastlines()

ax.add_feature(cfeature.LAND, facecolor='gainsboro', alpha=0.5)
# ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                               fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
index_box1 = mpatches.Rectangle((50, -10), 20, 20,
                                fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
index_box2 = mpatches.Rectangle((90, -10), 20, 10, 
                                fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
ax.add_patch(index_box)
ax.add_patch(index_box1)
ax.add_patch(index_box2)





gdf_onset['decade_start'] = (gdf_onset['year'] // 10) * 10
decades = sorted(gdf_onset['decade_start'].unique())
print(decades)

cmap = plt.get_cmap('rainbow', len(decades))
markers = ['o', 's', '^', 'v', '<', '>', 'd', 'p', 'h']


x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
max_year = gdf_onset['year'].max()
for i, dec in enumerate(decades):
    subset = gdf_onset[gdf_onset['decade_start'] == dec]
    end = min(dec + 9, max_year)              # cap at 2023
    label = f"{dec}-{end}"
    ax.scatter(
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


ax.text(0.095, 0.485, 'NINO3 region', transform=ax.transAxes, fontsize=10)
ax.text(0.75, 0.35, 'DMI regions',  transform=ax.transAxes, fontsize=10)


plt.subplots_adjust(right=0.8)  
ax.legend(
    title="Onset decade",
    fontsize=9,
    loc="center left",  
    bbox_to_anchor=(1.02, 0.5),  # x=1.02 just to the right of the axes, y=0.5 = middle
    borderaxespad=0.0,
    framealpha=0.7,
    markerscale=1.5,
    frameon=False
)

ax.text(0.05, 0.98, 'a', transform=ax.transAxes, fontsize=14, bbox=dict(
        boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
        facecolor='white',         # box fill color
        edgecolor='black',         # box edge color
        linewidth=0.5                # edge line width
    ))

plt.title('Spatio-temporal distribution of conflict onsets, n=555', fontsize=12)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_NINO3type2_psi_raw.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
