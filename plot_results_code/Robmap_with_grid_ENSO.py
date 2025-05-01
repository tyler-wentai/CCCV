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


# path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3_square2_wGeometry.csv'
# df = pd.read_csv(path)

# df['geometry'] = df['geometry'].apply(wkt.loads)

# # Create a GeoDataFrame, specifying the geometry column
# gdf = gpd.GeoDataFrame(df, geometry='geometry')

# # Optionally, set the coordinate reference system (CRS) if known, for example WGS84
# gdf.set_crs(epsg=4326, inplace=True)


# gdf_agg =gdf.groupby('loc_id').agg({
#     'geometry': 'first',
#     'psi': 'first',
#     'conflict_count':'sum',
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
# bounds = [0, 0.45, np.max(gdf_agg['psi'])]
# cmap = mcolors.ListedColormap(["blue", "gainsboro"])
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
#         'ticks': bounds
#     },
#     ax=ax,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax.coastlines()
# ax.add_patch(index_box)
# cbar = gdf_plot.get_figure().axes[-1]
# # cbar.set_yticklabels(['0%', '80%', '100%'])
# cbar.set_title("Teleconnection\nstrength\n(percentile)", fontsize=9)
# plt.title('NINO3 Teleconnection Group Paritioning', fontsize=10)
# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/RobMAP_square2_NINO3_percent.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()


####################################
####################################

path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_mrsosNINO3_square4_wGeometry.csv'
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
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# create a custom colormap
# bounds = [0, 1.4415020, np.max(gdf_agg['psi'])]
bounds = [np.min(gdf_agg['psi']), 0, np.max(gdf_agg['psi'])] #psi_quants
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

cmap = 'PRGn' #'gist_heat_r'

index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                        fill=True, facecolor='gray', edgecolor=None, linewidth=1.5, alpha=0.30,
                        transform=ccrs.PlateCarree())

gl.top_labels       = False 
ax.set_global()
gdf_plot = gdf_agg.plot(
    column='psi',    
    cmap=cmap, #'tab20c_r',
    norm=TwoSlopeNorm(vmin=bounds[0], vcenter=bounds[1], vmax=bounds[2]) ,
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
ax.add_patch(index_box)
cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '80%', '100%'])
cbar.set_title("Teleconnection", fontsize=9)
x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
ax.scatter(x, y, color='blue', s=1.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)
plt.title('ENSO (mrsos+NINO3) Teleconnection', fontsize=10)
plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_mrsosNINO3_psi_raw.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





###################################
###################################
# path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_NINO3_square4_wGeometry.csv'
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
# bounds = psi_quants
# cmap = mcolors.ListedColormap(["#648FFF", "#785EF0", "#FFB000", "#FE6100", "#DC267F"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

# index_box = mpatches.Rectangle((-150, -5), 60, 10, 
#                         fill=None, edgecolor='darkviolet', linewidth=1.5,
#                         transform=ccrs.PlateCarree())

# gl.top_labels       = False 
# ax.set_global()
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
#     ax=ax,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax.coastlines()
# ax.add_patch(index_box)
# cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
# plt.title('NINO3 Teleconnection', fontsize=11)
# plt.tight_layout()
# # plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_square4_NINO3_psi_equionsetcount.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()








###################################
###################################
# path1 = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets/Onset_Binary_Global_NINO3_square4_wGeometry.csv'
# path2 = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets/Onset_Binary_Global_NINO3_square2_cindexnosd_wGeometry.csv'
# df1 = pd.read_csv(path1)
# df2 = pd.read_csv(path2)

# df1['geometry'] = df1['geometry'].apply(wkt.loads)
# df2['geometry'] = df2['geometry'].apply(wkt.loads)

# gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
# gdf1.set_crs(epsg=4326, inplace=True)
# gdf_agg1 =gdf1.groupby('loc_id').agg({
#     'geometry': 'first',
#     'psi': 'first',
#     'conflict_binary':'sum',
# }).reset_index()
# gdf_agg1 = gpd.GeoDataFrame(gdf_agg1, geometry='geometry')
# gdf_agg1.set_crs(gdf1.crs, inplace=True)

# gdf2 = gpd.GeoDataFrame(df2, geometry='geometry')
# gdf2.set_crs(epsg=4326, inplace=True)
# gdf_agg2 =gdf2.groupby('loc_id').agg({
#     'geometry': 'first',
#     'psi': 'first',
#     'conflict_binary':'sum',
# }).reset_index()
# gdf_agg2 = gpd.GeoDataFrame(gdf_agg2, geometry='geometry')
# gdf_agg2.set_crs(gdf2.crs, inplace=True)



# # Define a polygon with lat/lon coordinates
# fig, axes = plt.subplots(
#     nrows=2, ncols=1,
#     figsize=(8, 8),
#     subplot_kw={'projection': ccrs.Robinson()}
# )

# for idx, (ax, gdf, title) in enumerate(zip(axes, [gdf_agg1, gdf_agg2], [r'$4^{\degree} \times 4^{\degree}$', r'$2^{\degree} \times 2^{\degree}$'])):
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
#     gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
#     gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
#     gl.xlabel_style = {'size': 8}
#     gl.ylabel_style = {'size': 8}
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     gl.top_labels = False
#     if idx == 0:
#         gl.bottom_labels = False  # Hide bottom labels for the first plot to avoid redundancy

#     ax.set_global()
#     gdf.plot(
#         legend=True,
#         color='gainsboro',
#         ax=ax,
#         transform=ccrs.PlateCarree()
#     )
#     ax.add_geometries(
#         gdf['geometry'],
#         crs=ccrs.PlateCarree(),
#         facecolor='none',
#         edgecolor='dimgrey',
#         linewidth=0.5
#     )
#     ax.coastlines()
#     ax.set_title(title, fontsize=11)

# plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_square4&square2_global.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()
