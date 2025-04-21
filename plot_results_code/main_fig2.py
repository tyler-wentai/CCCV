
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

### SUBPLOT A
#
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_NINO3_wGeometry.csv'
df = pd.read_csv(path)

df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)
gdf_A =gdf.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf_A = gpd.GeoDataFrame(gdf_A, geometry='geometry')
gdf_A.set_crs(gdf.crs, inplace=True)


crit_val = 1.80
conds = [
    (gdf_A['psi'] <  crit_val),
    (gdf_A['psi'] >= crit_val)
]
choices = ['gainsboro', 'red']

gdf_A['color'] = np.select(conds, choices, default='gainsboro')


### SUBPLOT B
#
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMI_wGeometry.csv'
df = pd.read_csv(path)

df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)
gdf_agg =gdf.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf_agg = gpd.GeoDataFrame(gdf_agg, geometry='geometry')
gdf_agg.set_crs(gdf.crs, inplace=True)
gdf_agg = gdf_agg.rename(columns={'psi': 'psi_dmi'})


#
path1 = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_ANI_wGeometry.csv'
df1 = pd.read_csv(path1)

df1['geometry'] = df1['geometry'].apply(wkt.loads)
gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
gdf1.set_crs(epsg=4326, inplace=True)
gdf1_aggB =gdf1.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf1_aggB = gpd.GeoDataFrame(gdf1_aggB, geometry='geometry')
gdf1_aggB.set_crs(gdf.crs, inplace=True)
gdf1_aggB = gdf1_aggB.rename(columns={'psi': 'psi_ani'})

gdf_B = gdf_agg.merge(gdf1_aggB, on='loc_id', how='left')
gdf_B = gdf_B.rename(columns={'geometry_x': 'geometry'})
gdf_B = gdf_B.drop(columns=['geometry_y'])

# corresponding colors
dmi_color = '#FFB000'
ani_color = '#DC267F'
mix_color = '#FE6100'

crit_val = 1.35
conds = [
    (gdf_B['psi_dmi']  < crit_val) & (gdf_B['psi_ani']  < crit_val),
    (gdf_B['psi_dmi']  > crit_val) & (gdf_B['psi_ani']  < crit_val),
    (gdf_B['psi_dmi']  < crit_val) & (gdf_B['psi_ani']  > crit_val),
    (gdf_B['psi_dmi']  > crit_val) & (gdf_B['psi_ani']  > crit_val)
]
choices = ['gainsboro', dmi_color, ani_color, mix_color]
gdf_B['color'] = np.select(conds, choices, default='gainsboro')



### SUBPLOT C
#
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_NINO3final_square4_wGeometry.csv'
df = pd.read_csv(path)

df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)
gdf_C =gdf.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf_C = gpd.GeoDataFrame(gdf_C, geometry='geometry')
gdf_C.set_crs(gdf.crs, inplace=True)


crit_val = 1.80
conds = [
    (gdf_C['psi'] <  crit_val),
    (gdf_C['psi'] >= crit_val)
]
choices = ['gainsboro', 'red']

gdf_C['color'] = np.select(conds, choices, default='gainsboro')

### SUBPLOT D
#
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_DMIfinal_square4_wGeometry.csv'
df = pd.read_csv(path)

df['geometry'] = df['geometry'].apply(wkt.loads)
gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)
gdf_agg =gdf.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf_agg = gpd.GeoDataFrame(gdf_agg, geometry='geometry')
gdf_agg.set_crs(gdf.crs, inplace=True)
gdf_agg = gdf_agg.rename(columns={'psi': 'psi_dmi'})


#
path1 = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_ANIfinal_square4_wGeometry.csv'
df1 = pd.read_csv(path1)

df1['geometry'] = df1['geometry'].apply(wkt.loads)
gdf1 = gpd.GeoDataFrame(df1, geometry='geometry')
gdf1.set_crs(epsg=4326, inplace=True)
gdf1_aggD =gdf1.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()
gdf1_aggD = gpd.GeoDataFrame(gdf1_aggD, geometry='geometry')
gdf1_aggD.set_crs(gdf.crs, inplace=True)
gdf1_aggD = gdf1_aggD.rename(columns={'psi': 'psi_ani'})

gdf_D = gdf_agg.merge(gdf1_aggD, on='loc_id', how='left')
gdf_D = gdf_D.rename(columns={'geometry_x': 'geometry'})
gdf_D = gdf_D.drop(columns=['geometry_y'])

# corresponding colors
dmi_color = '#FFB000'
ani_color = '#DC267F'
mix_color = '#FE6100'

crit_val = 1.35
conds = [
    (gdf_D['psi_dmi']  < crit_val) & (gdf_D['psi_ani']  < crit_val),
    (gdf_D['psi_dmi']  > crit_val) & (gdf_D['psi_ani']  < crit_val),
    (gdf_D['psi_dmi']  < crit_val) & (gdf_D['psi_ani']  > crit_val),
    (gdf_D['psi_dmi']  > crit_val) & (gdf_D['psi_ani']  > crit_val)
]
choices = ['gainsboro', dmi_color, ani_color, mix_color]
gdf_D['color'] = np.select(conds, choices, default='gainsboro')




######## PLOTTING
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
    2, 2,
    figsize=(10, 6.67),
    subplot_kw={'projection': ccrs.Robinson()}
)
fig.subplots_adjust(hspace=-0.4)

#nino3
index_box0 = mpatches.Rectangle(
    (-150, -5),
    60,
    10,
    fill=True, 
    facecolor='red', 
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree())
index_box00 = mpatches.Rectangle(
    (-150, -5),
    60,
    10,
    fill=True, 
    facecolor='red', 
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree())

#dmi
index_box1 = mpatches.Rectangle(
    (50, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 70E - 50E
    20,         # height: 10N - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
index_box11 = mpatches.Rectangle(
    (50, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 70E - 50E
    20,         # height: 10N - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
index_box2 = mpatches.Rectangle(
    (90, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 110E - 90E
    10,         # height: 0 - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
index_box22 = mpatches.Rectangle(
    (90, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 110E - 90E
    10,         # height: 0 - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
#ani
index_box3 = mpatches.Rectangle(
    (-20,  -3),    # lower-left corner: 20°W, 3°S
     20,           # width: 0°E minus (–20°W)
      6,           # height: 3°N minus (–3°S)
    fill=True,
    facecolor=ani_color,  # or whatever color you like
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
index_box33 = mpatches.Rectangle(
    (-20,  -3),    # lower-left corner: 20°W, 3°S
     20,           # width: 0°E minus (–20°W)
      6,           # height: 3°N minus (–3°S)
    fill=True,
    facecolor=ani_color,  # or whatever color you like
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)



# ─── Plot 1 ───
gl1 = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl1.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl1.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl1.xlabel_style = {'size': 8}
gl1.ylabel_style = {'size': 8}
gl1.xformatter = LONGITUDE_FORMATTER
gl1.yformatter = LATITUDE_FORMATTER
gl1.top_labels = False
ax1.set_global()
gdf_A.plot(
    color=gdf_A['color'],
    ax=ax1,
    transform=ccrs.PlateCarree(),
    linewidth=0
)
ax1.add_geometries(
    gdf_A['geometry'],
    crs=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='dimgrey',
    linewidth=0.5
)
ax1.coastlines()
ax1.add_patch(index_box0)
ax1.set_title("El Niño Southern Oscillation (ENSO)", fontsize=10, pad=3)


# ─── Plot 2 ───
gl2 = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl2.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl2.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl2.xlabel_style = {'size': 8}
gl2.ylabel_style = {'size': 8}
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER
gl2.top_labels = False
ax2.set_global()
gdf_B.plot(
    color=gdf_B['color'],
    ax=ax2,
    transform=ccrs.PlateCarree(),
    linewidth=0
)
ax2.add_geometries(
    gdf_B['geometry'],
    crs=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='dimgrey',
    linewidth=0.5
)
ax2.coastlines()
ax2.add_patch(index_box1)
ax2.add_patch(index_box2)
ax2.add_patch(index_box3)
ax2.set_title("Indian Ocean Dipole (IOD) & Atlantic Niño (AN)", fontsize=10, pad=3)


# ─── Plot 3 ───
gl3 = ax3.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl3.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl3.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl3.xlabel_style = {'size': 8}
gl3.ylabel_style = {'size': 8}
gl3.xformatter = LONGITUDE_FORMATTER
gl3.yformatter = LATITUDE_FORMATTER
gl3.top_labels = False
ax3.set_global()
gdf_C.plot(
    color=gdf_C['color'],
    ax=ax3,
    transform=ccrs.PlateCarree(),
    linewidth=0
)
ax3.add_geometries(
    gdf_C['geometry'],
    crs=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='dimgrey',
    linewidth=0.5
)
ax3.coastlines()
ax3.add_patch(index_box00)
# ax3.set_title("Map C", fontsize=10, pad=4)


# ─── Plot 4 ───
gl4 = ax4.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl4.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl4.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl4.xlabel_style = {'size': 8}
gl4.ylabel_style = {'size': 8}
gl4.xformatter = LONGITUDE_FORMATTER
gl4.yformatter = LATITUDE_FORMATTER
gl4.top_labels = False
ax4.set_global()
gdf_D.plot(
    color=gdf_D['color'],
    ax=ax4,
    transform=ccrs.PlateCarree(),
    linewidth=0
)
ax4.add_geometries(
    gdf_D['geometry'],
    crs=ccrs.PlateCarree(),
    facecolor='none',
    edgecolor='dimgrey',
    linewidth=0.5
)
ax4.coastlines()
ax4.add_patch(index_box11)
ax4.add_patch(index_box22)
ax4.add_patch(index_box33)
# ax4.set_title("Map D", fontsize=10, pad=4)


# after all of your plotting for ax1:
ax1.text(
    0.02, 1.00, "A",                # x, y in axis‐fraction coordinates
    transform=ax1.transAxes,        # use axis coordinate system
    ha="left", va="top",           # align the text
    fontsize=12, fontweight="bold"  # tweak size/weight as desired
)
ax2.text(0.02, 1.00, "B", transform=ax2.transAxes,
         ha="left", va="top", fontsize=12, fontweight="bold")
ax3.text(0.02, 1.00, "C", transform=ax3.transAxes,
         ha="left", va="top", fontsize=12, fontweight="bold")
ax4.text(0.02, 1.00, "D", transform=ax4.transAxes,
         ha="left", va="top", fontsize=12, fontweight="bold")


fig.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/CCCV_mainfig2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





sys.exit()


# Define a polygon with lat/lon coordinates
fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER


#dmi
index_box1 = mpatches.Rectangle(
    (50, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 70E - 50E
    20,         # height: 10N - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
index_box2 = mpatches.Rectangle(
    (90, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 110E - 90E
    10,         # height: 0 - (-10S)
    fill=True,
    facecolor=dmi_color,
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)
#ani
index_box3 = mpatches.Rectangle(
    (-20,  -3),    # lower-left corner: 20°W, 3°S
     20,           # width: 0°E minus (–20°W)
      6,           # height: 3°N minus (–3°S)
    fill=True,
    facecolor=ani_color,  # or whatever color you like
    edgecolor='k',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)

gl.top_labels       = False 
ax.set_global()
gdf_plot = gdf_B.plot(
    color=gdf_B['color'],            # ← pass your pre‐computed colors directly
    ax=ax,
    transform=ccrs.PlateCarree()
)
ax.add_geometries(gdf_B['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
ax.coastlines()
ax.add_patch(index_box1)
ax.add_patch(index_box2)
ax.add_patch(index_box3)
plt.title('Indian Ocean Dipole (DMI) Teleconnection Group Paritioning', fontsize=10)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_DMI_psi_percent.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()