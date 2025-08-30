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

print('\n\nSTART ---------------------\n')

####################################
####################################

# Load your two DataArrays
path1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_DMI_type2.nc'
path2 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_DMI_type2_GPCC.nc'
ds1 = xr.open_dataarray(path1)
ds2 = xr.open_dataarray(path2)

# Define the index box once
index_box1 = mpatches.Rectangle(
    (-150, -5), 60, 10,
    facecolor='white', edgecolor='white', alpha=0.3,
    transform=ccrs.PlateCarree()
)
index_box2 = mpatches.Rectangle(
    (50, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 70E - 50E
    20,         # height: 10N - (-10S)
    fill=True,
    facecolor='white',
    edgecolor='white',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)

index_box3 = mpatches.Rectangle(
    (90, -10),  # lower-left corner (longitude, latitude)
    20,         # width: 110E - 90E
    10,         # height: 0 - (-10S)
    fill=True,
    facecolor='white',
    edgecolor='white',
    linewidth=1.5,
    alpha=0.30,
    transform=ccrs.PlateCarree()
)

# Create figure with 2 stacked Robinson maps
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(8, 8),
    subplot_kw={'projection': ccrs.Robinson()},
    constrained_layout=True,
    gridspec_kw={'hspace': 0.1}
)

# PLOT 1
gl1 = ax1.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=0.4, color='dimgray'
)
gl1.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl1.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl1.xlabel_style = {'size': 9}
gl1.ylabel_style = {'size': 9}
gl1.xformatter = LONGITUDE_FORMATTER
gl1.yformatter = LATITUDE_FORMATTER
gl1.top_labels = False

ax1.set_global()
ax1.coastlines()
im1 = ds1.plot(
    ax=ax1,
    transform=ccrs.PlateCarree(),
    cmap='PuRd',
    add_colorbar=True,
    cbar_kwargs={'shrink': 0.6, 'pad': 0.02}
)
ax1.add_patch(index_box1)
ax1.set_title('DMI Teleconnection', fontsize=11)
cax1 = im1.colorbar.ax
cax1.set_title("Teleconnection\nstrength", fontsize=8)
ax1.text(0.05, 0.98, 'a', transform=ax1.transAxes, fontsize=14, bbox=dict(
            boxstyle='square,pad=0.3',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',         # box fill color
            edgecolor=None,         # box edge color
            linewidth=1                # edge line width
        ))

# PLOT 2
gl2 = ax2.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=0.4, color='dimgray'
)
gl2.xlocator = mticker.FixedLocator(range(-180, 181, 60))
gl2.ylocator = mticker.FixedLocator(range(-60, 91, 30))
gl2.xlabel_style = {'size': 9}
gl2.ylabel_style = {'size': 9}
gl2.xformatter = LONGITUDE_FORMATTER
gl2.yformatter = LATITUDE_FORMATTER
gl2.top_labels = False

ax2.set_global()
ax2.coastlines()
im2 = ds2.plot(
    ax=ax2,
    transform=ccrs.PlateCarree(),
    cmap='PuRd',
    add_colorbar=True,
    cbar_kwargs={'shrink': 0.6, 'pad': 0.02}
)
ax2.add_patch(index_box2)
ax2.add_patch(index_box3)
ax2.set_title('EI-DMI Teleconnection', fontsize=11)
cax2 = im2.colorbar.ax
cax2.set_title("Teleconnection\nstrength", fontsize=8)
ax2.text(0.05, 0.98, 'b', transform=ax2.transAxes, fontsize=14, bbox=dict(
            boxstyle='square,pad=0.3',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',         # box fill color
            edgecolor=None,         # box edge color
            linewidth=1                # edge line width
        ))


# plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_globalteleconnections_DMI_vs_EI-DMI.png', dpi=300, pad_inches=0.01)
plt.show()