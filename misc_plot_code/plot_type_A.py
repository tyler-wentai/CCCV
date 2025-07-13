import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime, timedelta
import xarray as xr
import seaborn as sns
import matplotlib.image as mpimg
import geopandas as gpd
from matplotlib.colors import ListedColormap
import regionmask

print('\n\nSTART ---------------------\n')

# psi1 = xr.open_dataarray('/Users/tylerbagwell/Desktop/spei6_ERA5_mon_194001-202212.nc')
# lat1 = psi1['lat'].values
# lon1 = psi1['lon'].values
# variable1 = psi1.values[:,:]
# vals1 = variable1.flatten()

# psi2 = xr.open_dataarray('/Users/tylerbagwell/Desktop/spei6_ERA5_mon_194001-202212.nc')
# lat2 = psi2['lat'].values
# lon2 = psi2['lon'].values
# variable2 = psi2.values[:,:]
# vals2 = variable2.flatten()


# import xarray as xr
# import matplotlib.pyplot as plt

# Load the datasets
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
gdf1 = gpd.read_file(path_land)
gdf2 = gpd.read_file(path_maritime_0)

ds = xr.open_dataset('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_mrsosEEI.nc')
var_name = list(ds.data_vars)[0]
da = ds[var_name]
# da = da.sum(dim='month')
land_regs = regionmask.defined_regions.natural_earth_v5_0_0.land_110

# this creates an integer mask: land cells get region IDs ≥0, ocean cells get −1
mask = land_regs.mask(da)

# keep only land
da_land = da.where(mask>=0)


# Plot using xarray's built-in map projection
fig = plt.figure(figsize=(12, 6))
ax = plt.axes()
da_land.plot.pcolormesh(
    ax=ax,
    cmap='PRGn',
    x='lon',
    y='lat',
    add_colorbar=True
)
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.15)
# gdf2.plot(ax=ax, edgecolor=None, color='white')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title(f'{var_name}: Teleconnection Strength')

plt.show()


sys.exit()





psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_DMI_cai_0d5.nc')
print(psi)
print(psi.attrs)

psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')
lat1 = psi['lat'].values
lon1 = psi['lon'].values
variable1 = psi.values[:,:]

#variable1 = psi.values[7,:,:]
# variable1 = np.sum(np.abs(psi), axis=0)

print(variable1.flatten())
# val = variable1.flatten()
plt.hist(variable1.flatten(), bins=50)
plt.show()

#####
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
gdf1 = gpd.read_file(path_land)
gdf2 = gpd.read_file(path_maritime_0)

fig, ax = plt.subplots(figsize=(10, 6.6))

reds = plt.cm.Reds(np.linspace(0, 1, 256))
reds[0] = [1, 1, 1, 1]  # Set the lowest color (for zero values) to white
custom_cmap = ListedColormap(reds)
variable1_masked = np.ma.masked_where(variable1 == 0, variable1)

maxval = np.max(variable1)
levels = np.arange(0,maxval,1)

c = ax.pcolormesh(lon1, lat1, variable1_masked, cmap=custom_cmap, shading='auto', vmin=1.5)
cc = ax.imshow(variable1_masked, cmap='Reds')
# gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.15)
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
# ax.set_title('Teleconnection strength, $\Psi$ (ENSO)')
# c.set_clim(0, 4)
cbar = fig.colorbar(cc, ax=ax, orientation='vertical', fraction=0.1, pad=0.1, shrink=0.6)
cbar.set_label(r"$\Psi$", rotation=0, fontsize=14)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])


# plt.savefig('/Users/tylerbagwell/Desktop/teleconnection_global_NINO3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()


sys.exit()


plt.figure(figsize=(8, 6))
sns.histplot(vals1, color='blue', label='airtemp + precip', kde=True, stat="density", bins=int(len(vals2)/2000), alpha=0.5)
sns.histplot(vals2, color='red', label='airtemp + soilw', kde=True, stat="density", bins=int(len(vals2)/2000), alpha=0.5)

plt.legend()
plt.xlabel(r'$\Psi^{DMI}$')
plt.title(r"$\Psi^{DMI}$ computed with 'precip' vs. 'soilw'")
# plt.savefig('plots/psi_DMI_histograms.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()



sys.exit()


fig, axs = plt.subplots(2, 1, figsize=(8, 7))
fig.suptitle(r'Global land-based teleconnection strength, $\Psi_{Callahan2023}^{NIN03}$', fontsize=16)

prop = len(vals1)/len(vals2)
levels = [0.0,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.6]

ax = axs[0]
c = ax.contourf(lon1, lat1, variable1, cmap='YlOrRd', levels=levels)
# gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)
ax.set_title(r'$\Psi$ based on air_temp + precip')

ax = axs[1]
sns.histplot(vals1, bins=int(len(vals1)/2000), stat='density', kde=True, ax=ax, color='r')
ax.set_xlabel(r'$\Psi^{NINO3}$')
ax.set_title(r'$\Psi$ based on air_temp + precip')

# ax = axs[0,1]
# c = ax.contourf(lon2, lat2, variable2, cmap='YlOrRd', levels=levels)
# # gdf2.plot(ax=ax, edgecolor=None, color='white')
# gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
# ax.set_xlim([-180.0, 180.0])
# ax.set_ylim([-90.0, +90.0])
# fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)
# ax.set_title(r'$\Psi$ based on air_temp + soilw')

# ax = axs[1,1]
# sns.histplot(vals2, bins=int(len(vals2)/2000), stat='density', kde=True, ax=ax, color='r')
# ax.set_xlabel(r'$\Psi^{NINO3}$')
# ax.set_title(r'$\Psi$ based on air_temp + soilw')

# plt.savefig('plots/psi_callahan_precip_vs_soilw.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()





