import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import sys

print('\n\nSTART ---------------------\n')

path_countries = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
gdf_countries = gpd.read_file(path_countries)

print(gdf_countries.columns)

#sys.exit()

# psi setup
psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_Hsiang2011.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')

lat = psi['lat']
lon = psi['lon']

print(psi.lon.values)
lon_grid, lat_grid = np.meshgrid(psi.lon.values, psi.lat.values)
points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]

gdf = gpd.GeoDataFrame({
    'geometry': points,
    'value': psi.values.flatten()
}, crs="EPSG:4326")

# gdf_with_countries = gpd.sjoin(gdf, gdf_countries, how="inner")
# print(gdf_with_countries.head())
# country_means = gdf_with_countries.groupby('SOVEREIGNT')['value'].mean().reset_index()
# print(country_means)

test = gdf_countries[gdf_countries['SOVEREIGNT'] == 'Afghanistan']
#print(test)

# test.plot()
# plt.show()

# print(gdf)

point = Point(64.1, 33.3)
gs = gpd.GeoSeries([point])
result = test.within(gs)
print(result)

# result = gdf.within(test)
# print(result)
# print(np.unique(result, return_counts=True))