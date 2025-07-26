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

print('\n\nSTART ---------------------\n')


# path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMI_wGeometry.csv'
# df = pd.read_csv(path)

# df['geometry'] = df['geometry'].apply(wkt.loads)

# # Create a GeoDataFrame, specifying the geometry column
# gdf = gpd.GeoDataFrame(df, geometry='geometry')

# # Optionally, set the coordinate reference system (CRS) if known, for example WGS84
# gdf.set_crs(epsg=4326, inplace=True)


# gdf_agg =gdf.groupby('loc_id').agg({
#     'geometry': 'first',
#     'psi': 'first',
#     'conflict_binary':'sum',
# }).reset_index()

# # Convert the aggregated DataFrame back into a GeoDataFrame and set the active geometry column
# gdf_agg = gpd.GeoDataFrame(gdf_agg, geometry='geometry')

# # Optionally, set the CRS using the CRS from the original GeoDataFrame
# gdf_agg.set_crs(gdf.crs, inplace=True)

# onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/GeoArmedConflictOnset_v1_CLEANED.csv'
# df_onset = pd.read_csv(onset_path)    
# gdf_onset = gpd.GeoDataFrame(
#     df_onset, 
#     geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
#     crs="EPSG:4326"
# )


# # Define a polygon with lat/lon coordinates
# fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
# gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
# gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
# gl.xlabel_style = {'size': 8}
# gl.ylabel_style = {'size': 8}
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER

# # create a custom colormap
# bounds = [0, 1.35, np.max(gdf_agg['psi'])]
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# index_box1 = mpatches.Rectangle(
#     (50, -10),  # lower-left corner (longitude, latitude)
#     20,         # width: 70E - 50E
#     20,         # height: 10N - (-10S)
#     fill=True,
#     facecolor='green',
#     edgecolor=None,
#     linewidth=1.5,
#     alpha=0.30,
#     transform=ccrs.PlateCarree()
# )

# index_box2 = mpatches.Rectangle(
#     (90, -10),  # lower-left corner (longitude, latitude)
#     20,         # width: 110E - 90E
#     10,         # height: 0 - (-10S)
#     fill=True,
#     facecolor='green',
#     edgecolor=None,
#     linewidth=1.5,
#     alpha=0.30,
#     transform=ccrs.PlateCarree()
# )

# gl.top_labels       = False 
# ax.set_global()
# gdf_plot = gdf_agg.plot(
#     column='psi',    
#     cmap=cmap, #'tab20c_r',
#     norm=norm,   
#     legend=True,                   
#     legend_kwds={
#         'label': "Weak group       Strong group",
#         'orientation': "vertical", 
#         'shrink': 0.6,
#         'ticks': bounds
#     },
#     ax=ax,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax.coastlines()
# ax.add_patch(index_box1)
# ax.add_patch(index_box2)
# cbar = gdf_plot.get_figure().axes[-1]
# # cbar.set_yticklabels(['0%', '80%', '100%'])
# cbar.set_title("Teleconnection\nstrength\n(percentile)", fontsize=9)
# x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
# ax.scatter(x, y, color='blue', s=1.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)
# plt.title('Indian Ocean Dipole (DMI) Teleconnection Group Paritioning', fontsize=10)
# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/RobMAP_DMI_psi_percent.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()


####################################
####################################

path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4_wGeometry.csv'
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


# Define a polygon with lat/lon coordinates
fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# create a custom colormap
# bounds = [0, 1.4415020, np.max(gdf_agg['psi'])]
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

cmap = 'gist_heat_r'

index_box1 = mpatches.Rectangle(
    (50, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 70E - 50E
    20,         # height: 10N - (-10S)
    fill=True,
    facecolor='green',
    edgecolor=None,
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)

index_box2 = mpatches.Rectangle(
    (90, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 110E - 90E
    10,         # height: 0 - (-10S)
    fill=True,
    facecolor='green',
    edgecolor=None,
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)

gl.top_labels       = False 
ax.set_global()
gdf_plot = gdf_agg.plot(
    column='psi',    
    cmap=cmap, #'tab20c_r', 
    legend=True,                   
    legend_kwds={
        'orientation': "vertical", 
        'shrink': 0.6
    },
    ax=ax,
    transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
)
ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
ax.coastlines()
ax.add_patch(index_box1)
ax.add_patch(index_box2)
cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '80%', '100%'])
cbar.set_title("Teleconnection\nstrength", fontsize=9)
plt.title('Indian Ocean Dipole (DMI) Teleconnection Strength', fontsize=10)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/RobMAP_DMI_psi_raw_type2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
