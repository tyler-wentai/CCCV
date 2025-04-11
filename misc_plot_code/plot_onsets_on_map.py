import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys

print('\n\nSTART ---------------------\n')


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
df = pd.read_csv(onset_path)


# Convert to GeoDataFrame using onset_lon and onset_lat
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.onset_lon, df.onset_lat),
    crs="EPSG:4326"
)

print(gdf.shape)

# Create a figure with the desired size
fig = plt.figure(figsize=(4.5, 3.5))

# Set up an axes with the Robinson projection
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()

# Fill the map with ocean color (this acts like the map boundary fill)
# ax.add_feature(cfeature.OCEAN, facecolor='#E0FFFB')

ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)

# Add land (continents) with the chosen color
ax.add_feature(cfeature.LAND, facecolor='silver') #'#E0ECBA'

# Draw coastlines with a specified linewidth
ax.coastlines(linewidth=0.75)

# Set up gridlines with custom tick locations
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
gl.xlocator = mticker.FixedLocator(range(-120, 121, 60))  # meridians every 60°
gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
gl.xlabel_style = {'size': 7}
gl.ylabel_style = {'size': 7}
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Adjust gridline label positions:
# - For parallels (latitude) we want labels on the left only.
# - For meridians (longitude) we want labels on the right.
# By default, Cartopy labels latitudes on the left and longitudes on the bottom.
# To mimic Basemap:
# gl.left_labels      = False
# gl.right_labels     = False   
# gl.bottom_labels    = True  
gl.top_labels       = False 

x, y = gdf['onset_lon'].values, gdf['onset_lat'].values
ax.scatter(x, y, color='red', s=0.2, marker='o', transform=ccrs.PlateCarree(), zorder=5)

# ------------------------------------------------------------------------------


# Transform the lat/lon coordinates to the map projection coordinates.
# This is similar to using Basemap's m() function.
points = ax.projection.transform_points(ccrs.PlateCarree(), x, y)
x = points[:, 0]
y = points[:, 1]

# Stack x and y to feed into the Gaussian KDE
xy = np.vstack([x, y])
kde = gaussian_kde(xy, bw_method=0.15)

# Create a grid over the extent of the projected points
xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi.reshape(xi.shape)

# Mask low-density values so they won't be plotted
density_threshold = np.percentile(zi, 90)  # adjust the percentile as needed
zi_masked = np.where(zi > density_threshold, zi, np.nan)
xi_masked = np.where(zi > density_threshold, xi, np.nan)
yi_masked = np.where(zi > density_threshold, yi, np.nan)

# Plot the density contours.
# Since xi, yi, and zi are in the projection coordinate system, no additional transform is needed.
# ax.contourf(xi_masked, yi_masked, zi_masked, levels=4, cmap='Greys', alpha=1)


plt.title('Onset of armed conflict, 1950-2023\nn=555', fontsize=8)

# Show plot
plt.savefig('/Users/tylerbagwell/Desktop/conflict_onset_map.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
