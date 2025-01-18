import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import pearsonr, spearmanr, linregress
import pandas as pd
import sys
from datetime import datetime, timedelta
import xarray as xr
from pingouin import partial_corr
import statsmodels.api as sm
from prepare_index import *
from pathlib import Path

print('\n\nSTART ---------------------\n')

start_year  = 1980
end_year    = 2023

file_path_sst = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc'
var_str = 'tp'

ds = xr.open_dataset(file_path_sst)

# change dates to time format:
dates = pd.to_datetime(ds['date'].astype(str), format='%Y%m%d')
ds = ds.assign_coords(date=dates)
ds = ds.rename({'date': 'valid_time'})

# Access longitude and latitude coordinates
lon1 = ds['longitude']
lat1 = ds['latitude']

lat_int_mask = (lat1 % 0.25 == 0)
lon_int_mask = (lon1 % 0.25 == 0)
ds = ds.sel(latitude=lat1[lat_int_mask], longitude=lon1[lon_int_mask])

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    longitude = ds['longitude']
    longitude = ((longitude + 180) % 360) - 180
    ds = ds.assign_coords(longitude=longitude)
    return ds

# Apply conversion if necessary
if lon1.max() > 180:
    ds = convert_longitude(ds)
ds = ds.sortby('longitude')

clim_ind = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
                          end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))


ts = xr.DataArray(
    clim_ind["ANOM"],
    coords=[clim_ind.index],   # use the pandas DatetimeIndex as the coords
    dims=["valid_time"]        # name the dimension 'time'
)

# sst_aligned, ts_aligned = xr.align(ds[var_str], ts, join="inner")
# print("sst_aligned shape:", sst_aligned.shape)
# print("ts_aligned shape:   ", ts_aligned.shape)
print("raw variable shape:", ds[var_str].shape)
n_time, n_lat, n_long = ds[var_str].shape


# print(ts_aligned)
# valid_time = ts_aligned['valid_time']
# month_mask = np.where()

# m1, m2, m3 = 12, 1, 2
# ts_time = pd.to_datetime(ts_aligned['valid_time'])  # Convert numpy datetime64 array to pandas DatetimeIndex
# mask = ts_time.month.isin([m1, m2, m3])

# ts_month = ts_aligned.sel(valid_time=ts_time[mask])

# sst_month, ts_month = xr.align(ds[var_str], ts_month, join="inner")


# valid_time = pd.to_datetime(ts_aligned['valid_time'])
# i_start = np.where(valid_time.month == 12)




# m1, m2, m3 = 12, 1, 2
# ts_time = pd.to_datetime(ts_aligned['valid_time'])
# mask = ts_time.month.isin([m1, m2, m3])

# index_avg = ts_aligned.sel(valid_time = ts_time[mask])


### NINO3.4
# 1) Add a 'DJF_year' column that treats December as belonging to the *next* year
clim_ind['year'] = clim_ind.index.year
clim_ind['month'] = clim_ind.index.month
clim_ind['DJF_year'] = clim_ind.index.year
clim_ind.loc[clim_ind.index.month == 12, 'DJF_year'] += 1

# 2) Filter for only DJF months (12, 1, 2)
djf = clim_ind[clim_ind.index.month.isin([12, 1, 2])]

# 3) Group by 'DJF_year' and compute the mean anomaly to obtain annualized index values
ann_ind = djf.groupby('DJF_year').ANOM.agg(['mean', 'count']).reset_index()
ann_ind = ann_ind[ann_ind['count'] == 3]    # Only keep years with all three months of data
ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'DJF_year': 'year'})
ann_ind = ann_ind.drop(['count'], axis=1)



#####
n_months = 12

corr_monthly = np.empty((n_months, n_lat, n_long))

def pearsonr_func(a, b):
    return pearsonr(a, b)

for i in range(1,n_months+1):
    print("\n...Starting compute of tropical month:", i, "of", n_months)
    if (i<=7): 
        m = i + 5
        y = 1
    else:
        m = i - 7
        y = 0

    # Convert "year" + month to a DatetimeIndex (assuming day=1)
    ann_help = ann_ind.copy()
    ann_help['date'] = pd.to_datetime({
        'year': ann_ind['year'] + y,        # Note: + y is here to help with computing corr's of months in the previous tropical year or present year
        'month': m,
        'day': 1
    })


    ann_help.set_index('date', inplace=True)
    ann_help.drop(columns='year', inplace=True)

    ann_ind_ts = xr.DataArray(
        ann_help["ann_ind"],
        coords=[ann_help.index],   # use the pandas DatetimeIndex as the coords
        dims=["valid_time"]        # name the dimension 'time'
    )

    var_aligned, ind_aligned = xr.align(ds[var_str], ann_ind_ts, join="inner")

    # standardize the align variable data
    mean_data           = var_aligned.mean(dim='valid_time', skipna=True)
    std_data            = var_aligned.std(dim='valid_time', skipna=True)
    var_standardized    = (var_aligned - mean_data) / std_data
    var_standardized    = var_standardized.where(std_data != 0, 0) # handle the special case: if std == 0 at a grid cell, set all times there to 0

    # compute correlations and their p-values
    corr_map, pval_map = xr.apply_ufunc(
        pearsonr_func,
        var_standardized,       # first input
        ind_aligned,            # second input
        input_core_dims=[["valid_time"], ["valid_time"]],   # dimension(s) over which to compute
        output_core_dims=[[], []],                          # correlation, p-value are scalars per grid
        vectorize=True,                                     # run function for each (lat, lon) point
    )

    # set all correlations to zero if its p-value is less than the threshold
    threshold = 0.05
    pval_mask = pval_map < threshold
    pval_mask = pval_mask.values

    corr_results = corr_map.values
    corr_results = np.where(pval_mask, corr_results, 0)

    # save to corr_final
    corr_monthly[(i-1),:,:] = corr_results





print(type(corr_monthly))

corr_final = np.abs(corr_monthly)
corr_final = np.sum(corr_final, axis=0)


psi_var = xr.DataArray(corr_final,
                       coords = {
                           "latitude":  ds['latitude'],
                           "longitude": ds['longitude']
                        },
                        dims = ["latitude", "longitude"]
                        )

print(psi_var)



# corr_map = xr.corr(sst_aligned, ts_aligned, dim="valid_time")

# print(corr_map)

# corr_map = np.abs(corr_map)

psi_var.plot(
    x="longitude", 
    y="latitude",
    cmap="Reds",  # a diverging colormap is nice for correlations
)
plt.title("Psi Map")
plt.show()


sys.exit()



