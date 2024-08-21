import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import rasterio

print('\n\nSTART ---------------------')

file_path1 = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc'
file_path2 = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'

## Open the NetCDF file
dataset1 = nc.Dataset(file_path1, mode='r')

## Print the dataset to see its structure
print(dataset1.variables.keys())



# # Open the ASCII Grid file
# with rasterio.open(file_path2) as dataset:
#     # Read the dataset's profile (metadata)
#     print(dataset.profile)

#     # Read the data
#     data = dataset.read(1)  # Read the first band

#     # Optionally, visualize the data using matplotlib
#     fig, ax = plt.subplots()
#     cax = ax.imshow(data, cmap='nipy_spectral_r', vmax=0.1e7)
#     ax.set_title('Raster Data')

#     # Create a color bar with the same height as the plot
#     cbar = fig.colorbar(cax, ax=ax, shrink=0.7)  # Adjust the 'shrink' parameter as needed

#     # Save the figure
#     plt.savefig("/Users/tylerbagwell/Desktop/global_pop_grid2005.png", bbox_inches='tight', dpi=200)

#     # Show the plot
#     plt.show()

# # Read the coordinate variables
# latitudes = dataset1.variables['lat'][:]
# longitudes = dataset1.variables['lon'][:]

# # Print the metadata
# print("NetCDF Metadata:")
# print("Latitudes:", latitudes)
# print("Longitudes:", longitudes)
# print("Latitude range:", latitudes.min(), "-", latitudes.max())
# print("Longitude range:", longitudes.min(), "-", longitudes.max())
# print("Latitude step:", latitudes[1] - latitudes[0])
# print("Longitude step:", longitudes[1] - longitudes[0])

# # Open the ASCII file
# with rasterio.open(file_path2) as src:
#     population_data = src.read(1)
#     transform = src.transform
#     width = src.width
#     height = src.height
#     crs = src.crs

#     # Calculate the coordinates
#     #x_coords, y_coords = src.xy(np.arange(height), np.arange(width))

#     # Print the metadata
#     print("-------------------")
#     temperature_data = dataset1.variables['air']#[:]
#     print('air shape:', temperature_data.shape)
#     print(".--------------------")
#     print('pop shape:', population_data.shape)
#     print("ASCII Metadata:")
#     print("Transform:", transform)
#     print("Width:", width)
#     print("Height:", height)
#     print("CRS:", crs)
#     lon_values = np.array([transform * (col, 0) for col in range(width)])[:, 0]
#     lat_values = np.array([transform * (0, row) for row in range(height)])[:, 1]
#     print('pop lat:', lat_values)
#     print('pop lon:', lon_values)
#     # print("X Coordinates range:", x_coords.min(), "-", x_coords.max())
#     # print("Y Coordinates range:", y_coords.min(), "-", y_coords.max())
#     # print("Pixel size:", transform[0], transform[4])


##### MODIFY

# dat = nc.Dataset(file_path1)

# VAR1=dat.variables['air']

# lat = dat.variables['lat'][:]
# lon = dat.variables['lon'][:]
# time = dat.variables['time'][:]

# #print(VAR1) # this line will print the dimensions of the array, VAR1, which we've saved.

# lat_90_index = np.where(lat == -90)[0]

# # latitude lower and upper index
# latbounds = [ -89.75 , +90.00]
# latli = min(np.argmin( np.abs(lat - latbounds[0]) ), np.argmin( np.abs(lat - latbounds[1]) ))
# latui = max(np.argmin( np.abs(lat - latbounds[0]) ), np.argmin( np.abs(lat - latbounds[1]) ))

# print(latli, latui)
# dat_mod = dat.variables['air'][: , latli:latui , :]
# print(dat_mod.shape)
# print(dat_mod.variables['lat'][:])

# dat_mod_np = np.array(dat_mod) # often we have to change NetCDF variables to Numpy Arrays before we can save them.
# np.save('/Users/tylerbagwell/Desktop/air.2m.mon.mean_modified.npy', dat_mod_np)

##### MODIFY first attempt GPT
## Open the original NetCDF file in read-only mode
# ds = nc.Dataset(file_path1, 'r')

# # Read the latitude and temperature data
# latitudes = ds.variables['lat'][:]
# temperatures = ds.variables['air'][:]

# # Identify the index where lat=90
# lat_90_index = np.where(latitudes == -90)[0]
# print(lat_90_index)

# # Remove the lat=90 data
# if len(lat_90_index) > 0:
#     lat_90_index = lat_90_index[0]
#     latitudes = np.delete(latitudes, lat_90_index)
#     temperatures = np.delete(temperatures, lat_90_index, axis=1)  # air(time, lat, lon)

# # Create a new NetCDF file
# new_nc_file = '/Users/tylerbagwell/Desktop/air.2m.mon.mean_modified.nc'
# new_ds = nc.Dataset(new_nc_file, 'w', format='NETCDF4')

# # Create dimensions in the new file
# new_ds.createDimension('time', ds.dimensions['time'].size)
# new_ds.createDimension('lat', len(latitudes))
# new_ds.createDimension('lon', ds.dimensions['lon'].size)

# # Create the latitude and longitude variables
# time_var = new_ds.createVariable('time', 'f4', ('time',))
# lat_var = new_ds.createVariable('lat', 'f4', ('lat',))
# lon_var = new_ds.createVariable('lon', 'f4', ('lon',))

# # Assign the latitude and longitude data
# time_var[:] = ds.variables['time'][:]
# lat_var[:] = latitudes
# lon_var[:] = ds.variables['lon'][:]

# # Copy the temperature variable
# temp_var = new_ds.createVariable('air', 'f4', ('time', 'lat', 'lon',))  # should be air(time, lat, lon)

# temp_var[:] = temperatures

# # Copy any other variables (e.g., time) if needed
# # Example: new_ds.createVariable('time', 'f4', ('time',))[:] = ds.variables['time'][:]

# # Add any necessary attributes
# time_var.units = ds.variables['time'].units
# lat_var.units = ds.variables['lat'].units
# lon_var.units = ds.variables['lon'].units
# temp_var.units = ds.variables['air'].units

# # Close the new dataset
# new_ds.close()
# ds.close()

# print("New NetCDF file created without lat=90.")




#######
# print('NEW --------')
# file_path1 = '/Users/tylerbagwell/Desktop/air.2m.mon.mean_modified.nc'

# # Open the NetCDF file
# dataset1 = nc.Dataset(file_path1, mode='r')

# # Print the dataset to see its structure
# print(dataset1.variables.keys())

# # Read the coordinate variables
# latitudes = dataset1.variables['lat'][:]
# longitudes = dataset1.variables['lon'][:]

# # Print the metadata
# print("NetCDF Metadata:")
# print("Latitudes:", latitudes)
# print("Longitudes:", longitudes)
# print("Latitude range:", latitudes.min(), "-", latitudes.max())
# print("Longitude range:", longitudes.min(), "-", longitudes.max())
# print("Latitude step:", latitudes[1] - latitudes[0])
# print("Longitude step:", longitudes[1] - longitudes[0])


#######


#======================================================================
# from mpl_toolkits.basemap import Basemap, shiftgrid

# plt.style.use('ggplot')
# fig = plt.figure()

# parallels = np.arange(-40,40,20)
# meridians = np.arange(-40,80,20)

# ax1 = fig.add_subplot(1,1,1)

# m = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360) # USA

# # labels = [left,right,top,bottom]
# m.drawparallels(parallels,labels=[False,True,True,False])
# m.drawmeridians(meridians,labels=[True,False,False,True])
# # draw costlines and coutries
# m.drawcoastlines(linewidth=1.5)
# m.drawcountries(linewidth=0.5)

# #m.fillcontinents(color='grey',lake_color='aqua')
# # compute the lons and lats to fit the projection
# x, y = m(*np.meshgrid(lon,lat))

# # draw filled contours.
# levels=np.arange(200,320,10) # this sets the colorbar levels
# vmin=200  # this sets the colorbar min
# vmax=320 # this sets the colorbar max
# ax1 = m.contourf(x,y,VAR1[1000,:,:],cmap=plt.cm.RdBu_r, levels=levels, vmin=vmin, vmax=vmax)

# # add colorbar.
# cbar = m.colorbar(ax1,location='bottom',pad="10%")
# font_size = 8 # Adjust as appropriate.
# cbar.ax.tick_params(labelsize=font_size)
# cbar.set_label('Air Temperature, (K)') # Python will respond to LaTeX syntax commands.

# #plt.title(r'Air Temperature, (K)')
# plt.show()






##### STEP 1: Detrend and Standardize
from scipy.signal import detrend
dat = nc.Dataset(file_path1)

VAR1=dat.variables['air']

lat = dat.variables['lat'][:]
lon = dat.variables['lon'][:]
time = dat.variables['time'][:]

from datetime import datetime, timedelta

# Define the reference date: 1800-01-01 00:00:00
reference_date = datetime(1800, 1, 1, 0, 0, 0)
target_date = datetime(1985, 1, 1, 0, 0, 0)

dates = np.array([reference_date + timedelta(hours=int(h)) for h in time])
start_time_ind = int(np.where(dates == target_date)[0])
print(np.where(dates == target_date)[0])
print(start_time_ind)
VAR1 = VAR1[start_time_ind:len(time), :, :]

# Initialize a new array to store the detrended data
VAR1_detrended = np.empty_like(VAR1)

# Get the shape of the data array
n_lat, n_long, n_time = VAR1.shape
print(n_lat, n_long, n_time)

# # Loop through each (lat, long) point and detrend the time series
# for i in range(n_lat):
#     for j in range(n_long):
#         print(i,j)
#         # Extract the time series at (lat, long) point
#         time_series = VAR1[:, i, j]
        
#         # Detrend the time series
#         detrended_series = detrend(time_series)
        
#         # Store the detrended time series back into the array
#         VAR1_detrended[:, i, j] = detrended_series


# xx = np.arange(start_time_ind, len(time), 1)
# plt.plot(xx, VAR1[:, 50, 50])
# plt.plot(xx, VAR1_detrended[:, 50, 50])
# plt.show()