import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt

print('\n\nSTART ---------------------\n')

lat_min=0.0; lat_max=20.0
lon_min=0.0; lon_max=20.0
stepsize = 1.0               # in degrees lat,lon

lat_center = (lat_max-lat_min)/2.0
lon_center = (lon_max-lon_min)/2.0

pnt = shapely.geometry.Point(lon_center, lat_center) # center point to form grid around

df = gpd.GeoDataFrame(geometry=[pnt], crs=4326)
# df = df.to_crs(df.estimate_utm_crs())
dist = 7000
coors = 2

xcenter = df.geometry.iloc[0].x
ycenter = df.geometry.iloc[0].y

xcoords = np.arange(start=xcenter-stepsize*(coors-1), stop=xcenter+stepsize*coors, step=stepsize)
ycoords = np.arange(start=ycenter-stepsize*(coors-1), stop=ycenter+stepsize*coors, step=stepsize)

coords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1,2)

centerpoints = gpd.points_from_xy(x=coords[:,0], y=coords[:,1])
squares = [p.buffer(distance=stepsize, cap_style=3) for p in centerpoints]
df2 = gpd.GeoDataFrame(geometry=squares, crs=df.crs)

# df = df.to_crs(4326)
# df2 = df2.to_crs(4326)
# ax = df.plot(color="red", markersize=200, figsize=(5, 5), zorder=1)
# df2.boundary.plot(ax=ax, zorder=0)
# plt.show()

# print(coords)




#############
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"

gdf1 = gpd.read_file(path_land)

countries_in = 'Zimbabwe'

gdf1 = gdf1[gdf1['SOVEREIGNT']==countries_in]
pnt = shapely.geometry.Point(gdf1.centroid.x, gdf1.centroid.y) # center point to form grid around

df = gpd.GeoDataFrame(geometry=[pnt], crs=4326)
# df = df.to_crs(df.estimate_utm_crs())
dist = 7000
coors = 2

xcenter = df.geometry.iloc[0].x
ycenter = df.geometry.iloc[0].y

stepsize = 1.0


print(gdf1.geometry)
coords = list(gdf1.geometry)

print(gdf1.geometry[0].exterior)

# Extract the y-coordinates (latitude values)
# latitudes = [coord[1] for coord in coords]

# # Get the minimum and maximum latitude
# min_latitude = min(latitudes)
# max_latitude = max(latitudes)

# xcoords = np.arange(start=xcenter-stepsize*(coors-1), stop=xcenter+stepsize*coors, step=stepsize)
# ycoords = np.arange(start=ycenter-stepsize*(coors-1), stop=ycenter+stepsize*coors, step=stepsize)

# print(min_latitude, max_latitude)