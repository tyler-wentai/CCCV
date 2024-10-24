import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
import xarray as xr
from matplotlib.colors import ListedColormap
import numpy as np

psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_Hsiang2011_nino3.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')
lat0 = psi['lat'].values
lon0 = psi['lon'].values
variable0 = psi.values[:,:]

psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_callahan_NINO3_0dot5_soilw.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')
lat = psi['lat'].values
lon = psi['lon'].values
variable1 = psi.values[:,:]

psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_callahan_DMI.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')
lat = psi['lat'].values
lon = psi['lon'].values
variable2 = psi.values[:,:]

psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_callahan_DMI.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')
lat = psi['lat'].values
lon = psi['lon'].values
variable3 = psi.values[:,:]

path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
gdf1 = gpd.read_file(path_land)
gdf2 = gpd.read_file(path_maritime_0)


#### PLOTTING
vmin = np.min([np.min(variable0),np.min(variable1),np.min(variable2),np.min(variable3)])
vmax = np.max([np.max(variable0),np.max(variable1),np.max(variable2),np.max(variable3)])
print(vmin, vmax)
levels=np.arange(0.2,+1.6,0.20) # this sets the colorbar levels
# colors = ['#ccdbfd', '#e2eafc', '#fff0f3','#ff8fa3', '#ff4d6d', '#a4133c', '#800f2f']
colors = ['#023e8a', '#0096c7', '#48cae4', '#caf0f8','#ffba08', '#e85d04', "#d00000", "#9d0208", '#6a040f', '#370617', '#03071e']

fig, axs = plt.subplots(2, 2, figsize=(9, 7))

fig.suptitle(r'Correlation $\rho$ of ONI and Air Temp.', fontsize=16)

ax = axs[0, 0]
c = ax.contourf(lon, lat, variable0, cmap='YlOrRd', levels=levels)
# gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ax.set_title('airtemp_month_lag=0')
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)

ax = axs[0, 1]
c = ax.contourf(lon, lat, variable1, cmap='YlOrRd')
# gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, vmin=0.5)
ax.set_title('airtemp_month_lag=1')
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)

ax = axs[1, 0]
c = ax.contourf(lon, lat, variable2, colors=colors, levels=levels)
gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ax.set_title('airtemp_month_lag=2')
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)

ax = axs[1, 1]
c = ax.contourf(lon, lat, variable3, colors=colors, levels=levels)
gdf2.plot(ax=ax, edgecolor=None, color='white')
gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
ax.set_title('airtemp_month_lag=3')
ax.set_xlim([-180.0, 180.0])
ax.set_ylim([-90.0, +90.0])
fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)

fig.tight_layout()
# plt.savefig('plots/rho_AMM_airtemp_NoOcean.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()