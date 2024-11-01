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


psi1 = xr.open_dataset('/Users/tylerbagwell/Desktop/psi_nino3_air_pm_2deg_maxmonthindex.nc')
lat1 = psi1['lat'].values
lon1 = psi1['lon'].values
variable1 = psi1['spi6'].values[400,:,:]
vals1 = variable1.flatten()

# psi2 = xr.open_dataset('/Users/tylerbagwell/Desktop/spei6_ERA5_mon_194001-202212.nc')
# lat2 = psi2['lat'].values
# lon2 = psi2['lon'].values
# variable2 = psi2['spei6'].values[0,:,:]
# vals2 = variable2.flatten()






path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
gdf1 = gpd.read_file(path_land)
gdf2 = gpd.read_file(path_maritime_0)

fig, ax = plt.subplots(figsize=(10, 6.6))
# fig.suptitle(r'Global land-based teleconnection strength, $\Psi^{DMI}$', fontsize=16)

# prop = len(vals1)/len(vals2)
levels = [-4.0,-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0]

c = ax.contourf(lon1, lat1, variable1, cmap='RdYlBu', levels=levels)
# gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)
ax.set_title(r'spi6-land, t400')

# plt.savefig('plots/spi6_land_global_t400.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
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





