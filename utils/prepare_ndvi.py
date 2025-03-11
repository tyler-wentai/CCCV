import matplotlib.pyplot as plt
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
import geopandas as gpd
import numpy as np

print('\n\nSTART ---------------------\n')


var1_path = '/Users/tylerbagwell/Desktop/cccv_data/ndvi_data/ndvi_annual_avg_1990.nc'
ds = xr.open_dataset(var1_path, mask_and_scale=False)

ndvi = ds['NDVI']

# Compute and print min and max values
ndvi_min = ndvi.min().item()
ndvi_max = ndvi.max().item()

print(f"NDVI min value: {ndvi_min}")
print(f"NDVI max value: {ndvi_max}")

ds['NDVI'].plot()
plt.title("NDVI from the AVHRR-Land Dataset")
plt.show()

# print(ds)
sys.exit()


import xarray as xr
import pandas as pd
import fsspec

year = 1990
base_url = f'https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/{year}/'
dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-01', freq='MS')
datasets = []


# Create an HTTPS filesystem object
fs = fsspec.filesystem('https')

for dt in dates:
    date_str = dt.strftime("%Y%m%d")
    # Build a pattern that uses a wildcard for the variable part of the filename
    # pattern = base_url + f'AVHRR-Land_v005_AVH13C1_NOAA-07_{date_str}_*.nc'   #1982
    pattern = base_url + f'AVHRR-Land_v005_AVH13C1_NOAA-11_{date_str}_*.nc'     #1990
    # pattern = base_url + f'VIIRS-Land_v001_NPP13C1_S-NPP_{date_str}_*.nc'     #2022
    # pattern = base_url + f'VIIRS-Land_v001_JP113C1_NOAA-20_{date_str}_*.nc'   #2023
    files = fs.glob(pattern)
    if files:
        file_url = files[0]
        try:
            # Open the remote file using fsspec and pass the file-like object to xarray
            with fs.open(file_url) as f:
                ds = xr.open_dataset(f, engine="h5netcdf", mask_and_scale=False)
                ds.load()
                datasets.append(ds)
                print(f"Loaded dataset for {date_str}")
        except Exception as e:
            print(f"Error loading dataset for {date_str}: {e}")
    else:
        print(f"NOTE: No file found for {date_str}!")

# Concatenate all datasets along the 'time' dimension
combined_ds = xr.concat(datasets, dim='time')
print(combined_ds)

annual_avg = combined_ds.mean(dim='time')



savepath = "/Users/tylerbagwell/Desktop/cccv_data/ndvi_data/ndvi_annual_avg_" + str(year) + ".nc"
annual_avg.to_netcdf(savepath)