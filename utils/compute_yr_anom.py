import numpy as np
from scipy.stats import linregress
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
from pathlib import Path

print('\n\nSTART ---------------------\n')

start_date  = '1950-01-01'
end_date    = '2023-12-31'

file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc' 
var1str = 'tp' 

ds1 = xr.open_dataset(file_path_VAR1)
ds1 = ds1.assign_coords(valid_time=ds1.valid_time.dt.floor('D'))
ds1 = ds1.rename({'valid_time': 'time'})

# Access longitude and latitude coordinates
lon1 = ds1['longitude']
lat1 = ds1['latitude']

resolution = 0.50

lat_int_mask = (lat1 % resolution == 0)
lon_int_mask = (lon1 % resolution == 0)
ds1 = ds1.sel(latitude=lat1[lat_int_mask], longitude=lon1[lon_int_mask])

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    longitude = ds['longitude']
    longitude = ((longitude + 180) % 360) - 180
    ds = ds.assign_coords(longitude=longitude)
    return ds

# Apply conversion if necessary
if lon1.max() > 180:
    ds1 = convert_longitude(ds1)

ds1 = ds1.sortby('longitude')

ds1 = ds1.assign_coords(
    longitude=np.round(ds1['longitude'], decimals=2),
    latitude=np.round(ds1['latitude'], decimals=2)
)

ds_sliced = ds1.sel(time=slice(start_date, end_date))
ds_sliced = ds_sliced.sel(time=ds_sliced.time.dt.month.isin(range(5,13))) # keep months may-dec

lon1 = ds_sliced['longitude']
lat1 = ds_sliced['latitude']

annual_sum = ds_sliced.groupby("time.year").sum(dim="time")
print(annual_sum)

annual_sum.to_netcdf("/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/ERA5_tp_YearlySumMayDec_0d50_19502023.nc")