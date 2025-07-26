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


# path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_NINO3_square4_wGeometry.csv'
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
# bounds = [0, 1.4415020, np.max(gdf_agg['psi'])]
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# index_box = mpatches.Rectangle((-150, -5), 60, 10, 
#                         fill=True, facecolor='green', edgecolor=None, linewidth=1.5, alpha=0.30,
#                         transform=ccrs.PlateCarree())

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
#         'ticks': [0, 1.4415020, np.max(gdf_agg['psi'])]
#     },
#     ax=ax,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax.coastlines()
# ax.add_patch(index_box)
# cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '80%', '100%'])
# cbar.set_title("Teleconnection\nstrength\n(percentile)", fontsize=9)
# plt.title('NINO3 Teleconnection Group Paritioning', fontsize=10)
# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_NINO3_psi_percent.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
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
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color='k')
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

cmap = 'PuRd'


gl.top_labels       = False 
ax.set_global()
gdf_plot = gdf_agg.plot(
    column='conflict_count',    
    cmap='turbo', 
    legend=True,                   
    legend_kwds={
        'orientation': "vertical", 
        'shrink': 0.6
    },
    ax=ax,
    transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
)
ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='gray', linewidth=0.5)
ax.coastlines(color='white', linewidth=0.5)
cbar = gdf_plot.get_figure().axes[-1]
cbar.set_title("Conflict onset\ncount", fontsize=9)
plt.title('Onset of armed conflict, 1950-2023\nn=555', fontsize=12)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_ConflictCounts_square4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
