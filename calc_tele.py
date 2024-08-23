import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

print('\n\nSTART ---------------------\n')

file_path_COUNTRIES = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
file_path_POP = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'
gdf_countries = gpd.read_file(file_path_COUNTRIES)

#print(gdf_countries.columns)



# psi setup
psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/psi_Hsiang2011.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')

lon_grid, lat_grid = np.meshgrid(psi.lon.values, psi.lat.values)
points_psi = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]

lat = psi['lat']
lon = psi['lon']

print(psi.lon.values)
lon_grid, lat_grid = np.meshgrid(psi.lon.values, psi.lat.values)
points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]

gdf = gpd.GeoDataFrame({
    'geometry': points,
    'value': psi.values.flatten()
}, crs="EPSG:4326")

gdf_with_countries = gpd.sjoin(gdf, gdf_countries, how="left", predicate="within")
gdf_with_countries = gdf_with_countries[gdf_with_countries['SOVEREIGNT'].notna()] # SOVEREIGNT is the column for countries
country_means = gdf_with_countries.groupby('SOVEREIGNT')['value'].mean().reset_index()
print(country_means)

plt.hist(country_means['value'])
plt.show()


# color = np.where(all['SOVEREIGNT'] == 'Afghanistan', "red", "gray")

# all.plot(markersize = 0.5, color = color)
# plt.show()

# print(gdf)

# point = Point(64.1, 33.3)
# gs = gpd.GeoSeries([point])
# result = test.within(gs)
# print(result)

# result = gdf.within(test)
# print(result)
# print(np.unique(result, return_counts=True))




#### ---- population weighting
## pop setup
# pop = rioxarray.open_rasterio(file_path_POP)

# lon_grid_pop, lat_grid_pop = np.meshgrid(pop.x.values, pop.y.values)
# points_pop = [Point(lon, lat) for lon, lat in zip(lon_grid_pop.flatten(), lat_grid_pop.flatten())]

# print(pop.y.values)

# gdf_pop = gpd.GeoDataFrame({
#     'geometry': points_pop,
#     'value': pop.values.flatten()
# }, crs="EPSG:4326")

# print(gdf_psi.crs)
# print(gdf_pop.crs)

# def compute_average_value(geometry, gdf, value_column, radius):
#     buffer = geometry.buffer(radius)                # Create a buffer around the point with the specified radius
#     neighbors = gdf[gdf.geometry.within(buffer)]    # Find all points in gdf that are within this buffer
#     if not neighbors.empty:                         # Calculate the average of the values in the specified column
#         return neighbors[value_column].mean()
#     else:
#         return None

# radius = 0.25*(1.1)  # Working in degrees lat and lon
# print(gdf_pop)
# gdf_psi['pop_average_value'] = gdf_psi.apply(lambda row: compute_average_value(row.geometry, gdf_pop, 'value', radius), axis=1)
# print("\n", gdf_psi)