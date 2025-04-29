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

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
import numpy as np
import regionmask

# --- load the data -----------------------------------------------------------
ds  = xr.open_dataset(
    '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_mrsosANI.nc'
)
da  = ds['__xarray_dataarray_variable__']                         # choose the field you want to plot
da   = da.squeeze()                    # drop length‑1 dimensions, if any

# If lat is descending, flip it so pcolormesh draws correctly
if np.all(np.diff(da.lat) < 0):
    da = da.sortby('lat')

land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
mask = land.mask(ds)                       # 0 = land, 1 = ocean

da_land = da.where(mask == 0)              # set ocean cells → NaN

da_nonzero = da_land.where(da_land != 0)
std_land = da_nonzero.std(dim=('lat', 'lon'), skipna=True).item()
print(std_land)


ani_color = 'gainsboro'  # or whatever color you like
index_box1 = mpatches.Rectangle(
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

# --- build the map -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5.),
                        subplot_kw={'projection': ccrs.Robinson()})

ax.set_title('Cumulative correlation of (MRSOS, Atlantic Niño Index JJA) over tropical year', fontsize=8.5, pad=7)

# nice gridlines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlabels_top   = False
gl.ylabels_right = False
gl.xlocator      = mticker.FixedLocator(range(-180, 181, 60))
gl.ylocator      = mticker.FixedLocator(range(-60,   91, 30))
gl.xlabel_style  = {'size': 8}
gl.ylabel_style  = {'size': 8}
gl.xformatter    = LONGITUDE_FORMATTER
gl.yformatter    = LATITUDE_FORMATTER

# same two‑class colouring you used before
bounds = [-4*std_land, 0, +4*std_land]
# cmap   = mcolors.ListedColormap(['gainsboro', 'red'])
# norm   = mcolors.BoundaryNorm(bounds, cmap.N)

# draw the grid with pcolormesh
mesh = ax.pcolormesh(
    da_land.lon, da_land.lat, da_land,
    cmap='PRGn',
    norm=TwoSlopeNorm(
        vmin=bounds[0], vcenter=bounds[1], vmax=bounds[2]
    ),
    transform=ccrs.PlateCarree()
)

ax.add_patch(index_box1)

# coastlines & extent
ax.coastlines()
ax.set_global()

# colour‑bar that matches the old style
cbar = fig.colorbar(
    mesh, ax=ax, orientation='vertical',
    shrink=0.5, pad=0.07)
cbar.set_label('← more dry               more wet →', fontsize=9)  # label size
cbar.ax.set_title('Cumulative corr.', pad=10, fontsize=8)
cbar.ax.tick_params(labelsize=7)                # tick‑label size
plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_mrsosANI_corr.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


sys.exit()