import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import sys

print('\n\nSTART ---------------------\n')

onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
df = pd.read_csv(onset_path)


# Convert to GeoDataFrame using onset_lon and onset_lat
gdf = gpd.GeoDataFrame(
    df, 
    geometry=gpd.points_from_xy(df.onset_lon, df.onset_lat),
    crs="EPSG:4326"
)



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Set figure size
plt.figure(figsize=(6, 4))

# Initialize Basemap with a global view
m = Basemap(projection='robin',lon_0=0,resolution='c')

# Draw map boundaries and fill continents
m.drawcoastlines(linewidth=0.75)
# m.drawcountries(linewidth=0.45)
m.fillcontinents(color='antiquewhite', lake_color='aliceblue')
m.drawmapboundary(fill_color='aliceblue')

parallels = range(-60, 91, 30)  # Latitude lines every 30 degrees
meridians = range(-180, 181, 60)  # Longitude lines every 60 degrees
m.drawparallels(parallels, labels=[1,0,0,0], fontsize=8, linewidth=0.4)
m.drawmeridians(meridians, labels=[0,0,0,1], fontsize=8, linewidth=0.4)

# Project points and plot them
x, y = m(gdf['onset_lon'].values, gdf['onset_lat'].values)
xy = np.vstack([x, y])
kde = gaussian_kde(xy)

# Create a grid for contouring
xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
zi = kde(np.vstack([xi.flatten(), yi.flatten()]))

# Reshape zi back to grid shape
zi = zi.reshape(xi.shape)

# Set very low density values to NaN (not plotted)
density_threshold = np.percentile(zi, 70)  # adjust percentile as needed (e.g., bottom 5%)
zi_masked = np.where(zi > density_threshold, zi, np.nan)
xi_masked = np.where(zi > density_threshold, xi, np.nan)
yi_masked = np.where(zi > density_threshold, yi, np.nan)

# Plot density contours (masked)
contour = m.contourf(xi_masked, yi_masked, zi_masked, levels=6, cmap='Greys', alpha=0.4)
contour = m.contour(xi_masked, yi_masked, zi_masked, levels=6, cmap='Greys', alpha=0.8, linewidths=0.5)

m.scatter(x, y, color='red', marker='o', s=0.2, alpha=1)

# Set title
plt.title('Conflict Onset Locations (1950-2023)', fontsize=11)

# Show plot
plt.savefig('/Users/tylerbagwell/Desktop/conflict_onset_map.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()
