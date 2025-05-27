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
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# create a custom colormap
# bounds = [0, 1.4415020, np.max(gdf_agg['psi'])]
bounds = [np.min(gdf_agg['psi']), np.max(gdf_agg['psi'])] #psi_quants
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)

cmap =  'gist_heat_r' #'gist_heat_r'#'PRGn' 

index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                        fill=True, facecolor='gray', edgecolor=None, linewidth=1.5, alpha=0.30,
                        transform=ccrs.PlateCarree())

gl.top_labels       = False 
ax.set_global()
gdf_plot = gdf_agg.plot(
    column='psi',    
    cmap=cmap, #'tab20c_r',
    # norm=TwoSlopeNorm(vmin=bounds[0], vmax=bounds[1]) ,
    legend=True,                   
    legend_kwds={
        "pad": 0.08,
        'orientation': "vertical", 
        'shrink': 0.65
    },
    ax=ax,
    transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
)
ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
ax.coastlines()
ax.add_patch(index_box)

# colorbar
# cbar = gdf_plot.get_figure().axes[-1]



cbar_ax = gdf_plot.get_figure().axes[-1]

# 3) compute histogram of your data
data = gdf_agg['psi'].values
vmin, vmax = cbar_ax.get_ybound()
bins = np.linspace(vmin, vmax, 16)
hist, edges = np.histogram(data, bins=bins)
centers = 0.5 * (edges[:-1] + edges[1:])



# 4) make a twin x-axis on the colorbar
hist_ax = cbar_ax.twiny()

pad = hist.max() * 0.125          # 10% of your max count
hist_ax.set_xlim(-pad, hist.max())

# 5) draw horizontal bars
hist_ax.barh(centers, hist,
             height=(edges[1] - edges[0]),
             align='center',
             color='dimgray',
             edgecolor='white',
             linewidth=0.5,
             alpha=0.4)

# 6) clean up the new axis
hist_ax.xaxis.set_ticks_position('bottom')
hist_ax.xaxis.set_label_position('bottom')
hist_ax.set_xlabel('Count')
hist_ax.spines['top'].set_visible(False)
hist_ax.spines['right'].set_visible(False)
hist_ax.spines['left'].set_visible(False)

cbar_ax.yaxis.set_ticks_position('left')
cbar_ax.yaxis.set_label_position('left')
cbar_ax.set_title("Teleconnection\nstrength", fontsize=9)


x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
ax.scatter(x, y, color='limegreen', s=1.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)
ax.set_title('NINO3 Teleconnection', fontsize=10)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_NINO3type2_psi_raw.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()