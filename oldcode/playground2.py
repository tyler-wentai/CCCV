import geopandas as gpd
from shapely.wkt import loads
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np

start_year = 1985
end_year = 2023


wkt_polygon = "POLYGON ((49.78434425015483 -15.069827835880368, 49.78434425015483 -20.069827835880368, 44.78434425015483 -20.069827835880368, 44.78434425015483 -15.069827835880368, 49.78434425015483 -15.069827835880368))"
polygon = loads(wkt_polygon)
gdf_polygon = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    longitude = ds['longitude']
    longitude = ((longitude + 180) % 360) - 180
    ds = ds.assign_coords(longitude=longitude)
    return ds

# LOAD IN ANOMALIZED CLIMATE DATA
file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_t2m_raw.nc' # air temperature 2 meter
file_path_VAR2 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc'  # total precipitation
ds1 = xr.open_dataset(file_path_VAR1)
ds2 = xr.open_dataset(file_path_VAR2)

# change dates to time format:
dates = pd.to_datetime(ds1['date'].astype(str), format='%Y%m%d')
ds1 = ds1.assign_coords(date=dates)
ds1 = ds1.rename({'date': 'time'})
ds1 = ds1.rio.write_crs("EPSG:4326")                    # Ensure the dataset has a CRS
ds1 = ds1.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))   # Select data within the specified year range

dates = pd.to_datetime(ds2['date'].astype(str), format='%Y%m%d')
ds2 = ds2.assign_coords(date=dates)
ds2 = ds2.rename({'date': 'time'})
ds2 = ds2.rio.write_crs("EPSG:4326")                    # Ensure the dataset has a CRS
ds2 = ds2.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))   # Select data within the specified year range

lon1 = ds1['longitude']
lon2 = ds2['longitude']
if lon1.max() > 180:
    ds1 = convert_longitude(ds1)
if lon2.max() > 180:
    ds2 = convert_longitude(ds2)
ds1 = ds1.sortby('longitude')
ds2 = ds2.sortby('longitude')
ds1 = ds1.assign_coords(
    longitude=np.round(ds1['longitude'], decimals=2),
    latitude=np.round(ds1['latitude'], decimals=2)
    )
ds2 = ds2.assign_coords(
    longitude=np.round(ds2['longitude'], decimals=2),
    latitude=np.round(ds2['latitude'], decimals=2)
    )


# ds_yearly1 = ds1.groupby('time.year').mean()             # Group data by year and compute annual mean
# ds_yearly2 = ds2.groupby('time.year').mean()             # Group data by year and compute annual mean

lats = ds1['latitude'].values
lons = ds1['longitude'].values

lon_grid, lat_grid = np.meshgrid(lons, lats)
lon_flat = lon_grid.flatten()
lat_flat = lat_grid.flatten()
df_points = pd.DataFrame({'longitude': lon_flat, 'latitude': lat_flat})
gdf_points = gpd.GeoDataFrame(df_points, geometry=gpd.points_from_xy(df_points.longitude, df_points.latitude))
gdf_points.crs = 'epsg:4326'

gdf_points['within_polygon'] = gdf_points.within(polygon)
points_within_polygon = gdf_points[gdf_points['within_polygon']]
within_mask_flat = gdf_points['within_polygon'].values
within_mask = within_mask_flat.reshape(lat_grid.shape)

masked_data = ds1['t2m'].where(within_mask)
spatial_mean = masked_data.mean(dim=['latitude', 'longitude'], skipna=True)
yearly_mean = spatial_mean.resample(time='YE').mean()

print(yearly_mean)




# gdf.plot()
# plt.show()