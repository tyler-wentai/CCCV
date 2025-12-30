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

path1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_mrsos_pos-NINO3.nc'
df1 = xr.open_dataarray(path1).sel(var="psi_pos")

path2 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_mrsos_neg-NINO3.nc'
df2 = xr.open_dataarray(path2).sel(var="psi_neg")

# If you still have a "month" dim, collapse it (adjust to mean/sum as you prefer)
df1_2d = df1.sum("month") if "month" in df1.dims else df1
df2_2d = df2.sum("month") if "month" in df2.dims else df2

# Common color scale (symmetric about 0 for easy comparison)
vabs = float(
    max(
        np.abs(df1_2d.min(skipna=True)).item(),
        np.abs(df1_2d.max(skipna=True)).item(),
        np.abs(df2_2d.min(skipna=True)).item(),
        np.abs(df2_2d.max(skipna=True)).item(),
    )
)
norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

cmap = cmocean.cm.curl_r  # or keep your string if it's registered: 'cmo.curl_r'

def setup_ax(ax):
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=0.4)
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(range(-60,  91, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_global()
    ax.coastlines()

    # NINO3-ish box from your example
    index_box = mpatches.Rectangle(
        (-150, -5), 60, 10,
        fill=True, facecolor='gray', edgecolor=None, linewidth=1.5, alpha=0.30,
        transform=ccrs.PlateCarree()
    )
    ax.add_patch(index_box)

fig, axs = plt.subplots(
    2, 1, figsize=(9, 8),
    subplot_kw={"projection": ccrs.Robinson()},
    constrained_layout=True
)

# Top: psi_pos
setup_ax(axs[0])
m1 = df1_2d.plot(
    ax=axs[0],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)
axs[0].set_title("psi_pos")

# Bottom: psi_neg
setup_ax(axs[1])
m2 = df2_2d.plot(
    ax=axs[1],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    add_colorbar=False
)
axs[1].set_title("psi_neg")

# One shared colorbar
cbar = fig.colorbar(m2, ax=axs, orientation="horizontal", fraction=0.05, pad=0.04)
cbar.set_label("Teleconnection")

plt.show()
