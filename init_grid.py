import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys

print('\n\nSTART ---------------------\n')


#############
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"

gdf1 = gpd.read_file(path_land)

countries_in = 'Nigeria' #'Zimbabwe' #Afghanistan

gdf1 = gdf1[gdf1['SOVEREIGNT']==countries_in]
pnt = shapely.geometry.Point(gdf1.centroid.x, gdf1.centroid.y) # center point to form grid around


df = gpd.GeoDataFrame(geometry=[pnt], crs=4326)
# df = df.to_crs(df.estimate_utm_crs())


lon_center = df.geometry.iloc[0].x
lat_center = df.geometry.iloc[0].y

stepsize = 1.0

lon_min = np.min(gdf1.geometry.get_coordinates()['x']); lon_max = np.max(gdf1.geometry.get_coordinates()['x'])
lat_min = np.min(gdf1.geometry.get_coordinates()['y']); lat_max = np.max(gdf1.geometry.get_coordinates()['y'])

lon_start   = lon_center - np.ceil(lon_center-lon_min)
lon_end     = lon_center + np.ceil(lon_max-lon_center)
lat_start   = lat_center - np.ceil(lat_center-lat_min)
lat_end     = lat_center + np.ceil(lat_max-lat_center)

xcoords = np.arange(start=lon_start, stop=lon_end, step=stepsize)
ycoords = np.arange(start=lat_start, stop=lat_end, step=stepsize)

coords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1,2)

centerpoints = gpd.points_from_xy(x=coords[:,0], y=coords[:,1])
squares = [p.buffer(distance=(stepsize/2), cap_style=3) for p in centerpoints]
df2 = gpd.GeoDataFrame(geometry=squares, crs=df.crs)

# print(df2.intersects(gdf1.geometry, align=True))


df2 = df2[df2.intersects(gdf1.geometry.iloc[0])]



df = df.to_crs(4326)
df2 = df2.to_crs(4326)
ax = df.plot(color="green", markersize=200, figsize=(5, 5), zorder=3)
df2.boundary.plot(ax=ax, zorder=2, color='red')
gdf1.plot(ax=ax, zorder=0)
plt.show()

# print(gdf1.geometry[0])
# print(df2.intersects(gdf1.geometry[0]))
