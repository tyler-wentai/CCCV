import numpy as np
from scipy.stats import linregress
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
from pingouin import partial_corr
from utils.calc_annual_index import *
from pathlib import Path
import warnings

print('\n\nSTART ---------------------\n')

import xarray as xr

clim_index = 'NINO3'

start_year  = 1950
end_year    = 2023 #note that spi only included observations up to 2022

file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc' 
var1str = 'tp' 



ds1 = xr.open_dataset(file_path_VAR1)
ds1 = ds1.assign_coords(valid_time=ds1.valid_time.dt.floor('D'))
ds1 = ds1.rename({'valid_time': 'time'})


# change dates to time format:
# ds1 = ds1.rename({'lat': 'latitude'})
# ds1 = ds1.rename({'lon': 'longitude'})

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

# load index data
clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
                        end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Eindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))

common_lon  = np.intersect1d(ds1['longitude'], ds1['longitude'])
common_lat  = np.intersect1d(ds1['latitude'], ds1['latitude'])
common_time = np.intersect1d(ds1['time'], clim_ind.index.to_numpy())

ds1_common      = ds1.sel(time=common_time, longitude=ds1['longitude'], latitude=ds1['latitude'])
clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]

var1_common = ds1_common[var1str]

# Check shapes
print("var1_common shape:", var1_common.shape)
print("clim_ind shape:   ", clim_ind_common.shape)

n_time, n_lat, n_long = var1_common.shape

# Verify that coordinates are identical
assert np.array_equal(var1_common['time'], clim_ind_common.index)


def detrend_then_standardize_monthly(data, israin: bool = False):
    """
    1. Remove the mean seasonal cycle (12-month climatology)
    2. Detrend each calendar-month slice separately
    3. Divide by the post-detrend sigma for that month
    """
    # --- set‑up -------------------------------------------------------------
    data   = np.asarray(data, dtype=float)
    n      = data.size
    months = np.arange(n) % 12            # 0...11

    # --- 1. climatology 
    clim_mean = np.array([data[months == m].mean() for m in range(12)])
    anom      = data - clim_mean[months]  # mean removed

    # --- 2. detrend each calendar month 
    df = pd.DataFrame({
        "anom" : anom,
        "month": months,
        "time" : np.arange(n, dtype=float)
    })

    def _remove_trend(group):
        if len(group) < 2: # safety for very short series
            return group["anom"]
        slope, intercept, *_ = linregress(group["time"], group["anom"])
        return group["anom"] - (slope * group["time"] + intercept)

    df["detr"] = (
        df.groupby("month", group_keys=False)
            .apply(_remove_trend, include_groups=False))

    # --- 3. standardize 
    sigma = np.array([df.loc[df["month"] == m, "detr"].std(ddof=0) for m in range(12)]) # population sigma

    if israin:
        sigma = np.where(sigma == 0, 1.0, sigma)   # keep dry points at 0

    standardized = df["detr"].values / sigma[months]
    return standardized.tolist()


anom_file1 = Path('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/tp_anom_ERA5_0d5_19502023_FINAL.npy')

if anom_file1.exists():
    print("Both anomaly field files exist. Skipping processing.")

    var1_std = np.load(anom_file1)
    # shape
    print(var1_std.shape)

else:
    print("Anomaly field files are missing. Proceeding with processing.")
    var1_std = np.empty_like(var1_common) # Initialize a new array to store the standardized data

    print("Standardizing and de-trending climate variable data...")
    for i in range(n_lat):
        if (i%10==0): 
            print("...", i)
        for j in range(n_long):
            has_nan = np.isnan(var1_common[:, i, j]).any()
            if (has_nan==False):
                var1_std[:, i, j] = detrend_then_standardize_monthly(var1_common[:, i, j], israin=True)
            else: 
                var1_std[:, i, j] = var1_std[:, i, j]

    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/tp_anom_ERA5_0d5_' + str(start_year) + str(end_year) + '.npy', var1_std)

var1_monthly_array = xr.DataArray(data = var1_std,
                            coords={
                            "time": common_time,    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["time", "lat", "lon"],
                        attrs=dict(
                            description="Gridded monthly ERA5 tp anomaly data.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

path1_str = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/tp_anom_ERA5_0d5_19502023_wTimeLatLon.nc'
var1_monthly_array.to_netcdf(path1_str)