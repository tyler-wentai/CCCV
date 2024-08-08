import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime

file_path1 = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc'
file_path2 = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'

# Open the NetCDF file
dataset1 = nc.Dataset(file_path1, mode='r')

# Print the dataset to see its structure
print(dataset1.variables.keys())


import rasterio

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

# Read the coordinate variables
latitudes = dataset1.variables['lat'][:]
longitudes = dataset1.variables['lon'][:]

# Print the metadata
print("NetCDF Metadata:")
print("Latitudes:", latitudes)
print("Longitudes:", longitudes)
print("Latitude range:", latitudes.min(), "-", latitudes.max())
print("Longitude range:", longitudes.min(), "-", longitudes.max())
print("Latitude step:", latitudes[1] - latitudes[0])
print("Longitude step:", longitudes[1] - longitudes[0])

# Open the ASCII file
with rasterio.open(file_path2) as src:
    population_data = src.read(1)
    transform = src.transform
    width = src.width
    height = src.height
    crs = src.crs

    # Calculate the coordinates
    #x_coords, y_coords = src.xy(np.arange(height), np.arange(width))

    # Print the metadata
    print("--------------------")
    temperature_data = dataset1.variables['air'][:]
    print(temperature_data.shape)
    print(population_data.shape)
    print("ASCII Metadata:")
    print("Transform:", transform)
    print("Width:", width)
    print("Height:", height)
    print("CRS:", crs)
    # print("X Coordinates range:", x_coords.min(), "-", x_coords.max())
    # print("Y Coordinates range:", y_coords.min(), "-", y_coords.max())
    # print("Pixel size:", transform[0], transform[4])