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
import xarray as xr
import cmocean

print('\n\nSTART ---------------------\n')

####################################
####################################

path = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_mrsos_neg-NINO3.nc'
df = xr.open_dataarray(path)

print(df)
df = df.sel(var="psi_neg")

# print(df.sel(var="psi_pos"))
# sys.exit()
# df = df.sum(dim="month") 


# onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/GeoArmedConflictOnset_v1_CLEANED.csv'
# df_onset = pd.read_csv(onset_path)    
# gdf_onset = gpd.GeoDataFrame(
#     df_onset, 
#     geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
#     crs="EPSG:4326"
# )


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
bounds = [df.min(), df.max()] #psi_quants
# bounds = [df.min(), 0, df.max()] #psi_quants
# cmap = mcolors.ListedColormap(["gainsboro", "red"])
# norm = mcolors.BoundaryNorm(bounds, cmap.N)



cmap = 'cmo.curl_r'#'PRGn', 'PuOr'

index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                        fill=True, facecolor='gray', edgecolor=None, linewidth=1.5, alpha=0.30,
                        transform=ccrs.PlateCarree())

gl.top_labels       = False 
ax.set_global()
gdf_plot = df.plot(
    cmap=cmap, #'tab20c_r',
    # norm=TwoSlopeNorm(vmin=bounds[0], vcenter=bounds[1], vmax=bounds[2]) ,
    add_colorbar=True,
    ax=ax,
    transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
)
# ax.add_geometries(df['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
ax.coastlines()
ax.add_patch(index_box)
cbar = gdf_plot.get_figure().axes[-1]
# cbar.set_yticklabels(['0%', '80%', '100%'])
cbar.set_title("Teleconnection", fontsize=9)


plt.title('DMI Semi', fontsize=10)
plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_psi_NINO3_type2_semi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

