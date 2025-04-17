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

import xarray as xr

clim_index = 'NINO3'

start_year  = 1950
end_year    = 2023

file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_t2m_raw.nc' # air temperature 2 meter
file_path_VAR2 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc'  # total precipitation


ds1 = xr.open_dataset(file_path_VAR1)
ds2 = xr.open_dataset(file_path_VAR2)

ds1 = ds1.assign_coords(valid_time=ds1.valid_time.dt.floor('D'))
ds2 = ds2.assign_coords(valid_time=ds2.valid_time.dt.floor('D'))

# change dates to time format:
# dates = pd.to_datetime(ds1['date'].astype(str), format='%Y%m%d')
# ds1 = ds1.assign_coords(date=dates)
ds1 = ds1.rename({'valid_time': 'time'})

# dates = pd.to_datetime(ds2['date'].astype(str), format='%Y%m%d')
# ds2 = ds2.assign_coords(date=dates)
ds2 = ds2.rename({'valid_time': 'time'})

var1str = 't2m'
var2str = 'tp'

# Access longitude and latitude coordinates
lon1 = ds1['longitude']
lat1 = ds1['latitude']
lon2 = ds2['longitude']
lat2 = ds2['latitude']

resolution = 0.5

lat_int_mask = (lat1 % resolution == 0)
lon_int_mask = (lon1 % resolution == 0)
ds1 = ds1.sel(latitude=lat1[lat_int_mask], longitude=lon1[lon_int_mask])
ds2 = ds2.sel(latitude=lat2[lat_int_mask], longitude=lon2[lon_int_mask])

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    longitude = ds['longitude']
    longitude = ((longitude + 180) % 360) - 180
    ds = ds.assign_coords(longitude=longitude)
    return ds

# Apply conversion if necessary
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

common_lon  = np.intersect1d(ds1['longitude'], ds2['longitude']) #probably should check that this is not null
common_lat  = np.intersect1d(ds1['latitude'], ds2['latitude'])
common_time = np.intersect1d(ds1['time'], ds2['time'])
common_time = np.intersect1d(common_time, clim_ind.index.to_numpy())

ds1_common      = ds1.sel(time=common_time, longitude=common_lon, latitude=common_lat)
ds2_common      = ds2.sel(time=common_time, longitude=common_lon, latitude=common_lat)
clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]

var1_common = ds1_common[var1str]
var2_common = ds2_common[var2str]

# Check shapes
print("var1_common shape:", var1_common.shape)
print("var2_common shape:", var2_common.shape)
print("clim_ind shape:   ", clim_ind_common.shape)

n_time, n_lat, n_long = var1_common.shape

# Verify that coordinates are identical
assert np.array_equal(var1_common['longitude'], var2_common['longitude'])
assert np.array_equal(var1_common['latitude'], var2_common['latitude'])
assert np.array_equal(var1_common['time'], var2_common['time'])
assert np.array_equal(var1_common['time'], clim_ind_common.index)
assert np.array_equal(var2_common['time'], clim_ind_common.index)


def standardize_and_detrend_monthly(data, israin=False):
    data = np.array(data)
    n = len(data)
    months = np.arange(n) % 12  # Assign month indices 0-11
    means = np.array([data[months == m].mean() for m in range(12)])
    stds = np.array([data[months == m].std() for m in range(12)])

    if israin==False:
        standardized = (data - means[months]) / stds[months]
    else:
        stds = np.where(stds == 0, 1, stds) # Just making sure we do not divide by zero here, dividing by 1 won't affect a gridpoint that experience no rain anyway, still will be 0.
        standardized = (data - means[months]) / stds[months]

    data = standardized.tolist()
    n = len(data)
    df = pd.DataFrame({
        'value': data,
        'month': np.arange(n) % 12,  # Assign months 0-11
        'time': np.arange(n)         # Time index
    })
    
    def remove_trend(group):
        if len(group) < 2:
            return group['value']
        slope, intercept, _, _, _ = linregress(group['time'], group['value'])
        return group['value'] - (slope * group['time'] + intercept)
    
    # Apply detrending per month
    df['detrended'] = df.groupby('month').apply(remove_trend, include_groups=False).reset_index(level=0, drop=True)
    return df['detrended'].tolist()

def standardize_monthly(data, israin=False):
    data = np.array(data)
    n = len(data)
    months = np.arange(n) % 12  # Assign month indices 0-11

    # Calculate monthly means and standard deviations
    means = np.array([data[months == m].mean() for m in range(12)])
    stds = np.array([data[months == m].std() for m in range(12)])

    if not israin:
        standardized = (data - means[months]) / stds[months]
    else:
        # Avoid division by zero for months with no rainfall
        stds = np.where(stds == 0, 1, stds)
        standardized = (data - means[months]) / stds[months]

    return standardized.tolist()


anom_file1 = Path('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/t2m_anom_ERA5_0d5' + str(start_year) + str(end_year) + '.npy')
anom_file2 = Path('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/tp_anom_ERA5_0d5' + str(start_year) + str(end_year) + 'npy')

if anom_file1.exists() and anom_file2.exists():
    print("Both anomaly field files exist. Skipping processing.")

    var1_std = np.load(anom_file1)
    var2_std = np.load(anom_file2)
    # shape
    print(var1_std.shape)
    print(var2_std.shape)

else:
    print("One or both anomaly field files are missing. Proceeding with processing.")
    var1_std = np.empty_like(var1_common) # Initialize a new array to store the standardized data
    var2_std = np.empty_like(var2_common) # Initialize a new array to store the standardized data

    # var1_std = var1_common.values       # DELETE LATER!!!!!!!!!
    # var2_std = var2_common.values       # DELETE LATER!!!!!!!!!

    print("Standardizing and de-trending climate variable data...")
    for i in range(n_lat):
        if (i%10==0): 
            print("...", i)
        for j in range(n_long):
            var1_std[:, i, j] = standardize_and_detrend_monthly(var1_common[:, i, j])
            has_nan = np.isnan(var2_common[:, i, j]).any()
            if (has_nan==False):
                var2_std[:, i, j] = standardize_and_detrend_monthly(var2_common[:, i, j], israin=True)
            else: 
                var2_std[:, i, j] = var2_common[:, i, j]

    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/t2m_anom_ERA5_0d5' + str(start_year) + str(end_year) + '.npy', var1_std)
    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/tp_anom_ERA5_0d5' + str(start_year) + str(end_year) + '.npy', var2_std)


# Compute the annualized index value:
clim_ind_common.index = pd.to_datetime(clim_ind_common.index)     # Ensure 'date' to datetime and extract year & month
clim_ind_common = clim_ind_common.copy()
clim_ind_common['year'] = clim_ind_common.index.year
clim_ind_common['month'] = clim_ind_common.index.month

## --- NINO3
dec_df = clim_ind_common[clim_ind_common['month'] == 12].copy() # prepare December data from previous year
dec_df['year'] = dec_df['year'] + 1  # Shift to next year
dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

jan_feb_df = clim_ind_common[clim_ind_common['month'].isin([1, 2])].copy() # prepare January and February data for current year
jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

yearly['avg_ANOM'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

# may_to_dec_df = ENSO_ind_common[ENSO_ind_common['month'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])].copy() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
# index_AVG = may_to_dec_df.groupby('year')['ANOM'].mean().reset_index() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
# index_AVG = index_AVG.rename(columns={'ANOM': 'avg_ANOM'}) # DELETE !!!!!!!!!!!!!!!!!!!!!!!!

## --- NINO34
# dec_df = clim_ind_common[clim_ind_common['month'] == 12].copy() # prepare December data from previous year
# dec_df['year'] = dec_df['year'] + 1  # Shift to next year
# dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

# jan_feb_df = clim_ind_common[clim_ind_common['month'].isin([1, 2])].copy() # prepare January and February data for current year
# jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
# feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

# yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
# yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

# yearly['avg_ANOM'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
# index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

## --- DMI
# sep_oct_nov_df = clim_ind_common[clim_ind_common['month'].isin([9, 10, 11])].copy() # prepare January and February data for current year
# sep     = sep_oct_nov_df[sep_oct_nov_df['month'] == 9][['year', 'ANOM']].rename(columns={'ANOM': 'SEP_ANOM'})
# oct     = sep_oct_nov_df[sep_oct_nov_df['month'] == 10][['year', 'ANOM']].rename(columns={'ANOM': 'OCT_ANOM'})
# nov     = sep_oct_nov_df[sep_oct_nov_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})

# yearly = pd.merge(sep, oct, on='year', how='inner') # merge December, January, and February data
# yearly = pd.merge(yearly, nov, on='year', how='inner') # merge December, January, and February data

# yearly['avg_ANOM'] = yearly[['SEP_ANOM', 'OCT_ANOM', 'NOV_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
# index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

## --- ANI
# jun_jul_aug_df = clim_ind_common[clim_ind_common['month'].isin([6, 7, 8])].copy() # prepare June, July, August (JJA) data for current year
# jun     = jun_jul_aug_df[jun_jul_aug_df['month'] == 6][['year', 'ANOM']].rename(columns={'ANOM': 'JUN_ANOM'})
# jul     = jun_jul_aug_df[jun_jul_aug_df['month'] == 7][['year', 'ANOM']].rename(columns={'ANOM': 'JUL_ANOM'})
# aug     = jun_jul_aug_df[jun_jul_aug_df['month'] == 8][['year', 'ANOM']].rename(columns={'ANOM': 'AUG_ANOM'})

# yearly = pd.merge(jun, jul, on='year', how='inner') # merge June, July, August data
# yearly = pd.merge(yearly, aug, on='year', how='inner') # merge June, July, August data

# yearly['avg_ANOM'] = yearly[['JUN_ANOM', 'JUL_ANOM', 'AUG_ANOM']].mean(axis=1) # Calculate the average JJA ANOM value
# index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

####
n_months = 12 # NINO3
# n_months = 9 # DMI
# n_months = 10 # ANI

corrs_array_1 = np.empty((n_months,n_lat,n_long))
pvals_array_1 = np.empty((n_months,n_lat,n_long))
corrs_array_2 = np.empty((n_months,n_lat,n_long))
pvals_array_2 = np.empty((n_months,n_lat,n_long))
psi = np.empty((n_lat,n_long))

# index_dat['avg_ANOM_ENSO'] = index_dat['avg_ANOM_ENSO'].shift(-1) # NEED TO TEST WHICH YEAR OF ENSO DJF TO CORRELATE!!!!
# index_dat = index_dat.dropna(subset=['avg_ANOM_ENSO'])

print("\nComputing psi array...")
for i in range(n_lat):
    if (i%10==0): 
        print("...", i)
    for j in range(n_long):
        current_vars = pd.DataFrame(data=var1_std[:,i,j],
                                            index=var1_common['time'], #need to use var1_common since it still contains the time data
                                            columns=[var1str])
        current_vars[var2str] = np.array(var2_std[:,i,j])
        current_vars.index = pd.to_datetime(current_vars.index)
        current_vars['year'] = current_vars.index.year
        current_vars['month'] = current_vars.index.month

        # iterate through the months
        for k in range(1,13,1):
            # may-dec of year t
            if (k<=8):
                var_ts = current_vars[current_vars['month'] == int(k+4)].copy()
            else:
                var_ts = current_vars[current_vars['month'] == int(k-8)].copy()
                var_ts['year'] = var_ts['year'] - 1  # Shift to previous year

        # for k in range(1,10,1):
        #     #var_ts = current_vars[current_vars['month'] == int(k+4)].copy()
        #     # may-dec of year t
        #     if (k<=8):
        #         var_ts = current_vars[current_vars['month'] == int(k+4)].copy()
        #     else:
        #         var_ts = current_vars[current_vars['month'] == int(k-8)].copy()
        #         var_ts['year'] = var_ts['year'] - 1  # Shift to previous year

        # for k in range(1,11,1):
        #     #var_ts = current_vars[current_vars['month'] == int(k+4)].copy()
        #     # may-dec of year t
        #     var_ts = current_vars[current_vars['month'] == int(k+2)].copy()

            # compute correlations of yearly month, k, air anomaly with index 
            var_ts = pd.merge(var_ts, index_AVG, how='inner', on='year')

            has_nan = var_ts[[var1str, var2str, 'avg_ANOM']].isna().any().any()
            if not has_nan:
                corr_1 = partial_corr(data=var_ts, x=var1str, y='avg_ANOM', covar=var2str)
                # corr_2 = partial_corr(data=var_ts, x=var2str, y='avg_ANOM', covar=var1str)
                corr_2 = partial_corr(data=var_ts, x=var2str, y='avg_ANOM')

                corrs_array_1[int(k-1),i,j] = corr_1['r'].values[0]
                corrs_array_2[int(k-1),i,j] = corr_2['r'].values[0]
                pvals_array_1[int(k-1),i,j] = corr_1['p-val'].values[0]
                pvals_array_2[int(k-1),i,j] = corr_2['p-val'].values[0]

            else:
                corrs_array_1[int(k-1),i,j] = np.nan
                corrs_array_2[int(k-1),i,j] = np.nan
                pvals_array_1[int(k-1),i,j] = 1.
                pvals_array_2[int(k-1),i,j] = 1.

        corrs1 = pd.Series(corrs_array_1[:,i,j])
        corrs2 = pd.Series(corrs_array_2[:,i,j])
        pvals1 = pd.Series(pvals_array_1[:,i,j])
        pvals2 = pd.Series(pvals_array_2[:,i,j])

        has_nan = corrs1.isna().any()
        if has_nan==False:
            # var1
            corr1_total_sum = np.abs(corrs1[pvals1 < 0.05]).sum()
            # var2
            corr2_total_sum = np.abs(corrs2[pvals2 < 0.05]).sum()
            # corr2_total_sum = corrs2[pvals2 < 0.05].sum()
            # compute teleconnection (psi)
            psi[i,j] = corr1_total_sum + corr2_total_sum
            # psi[i,j] = corr2_total_sum
        else:
            psi[i,j] = np.nan

psi_array = xr.DataArray(data = psi,
                            coords={
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["lat", "lon"],
                        attrs=dict(
                            description="Psi, teleconnection strength inspired by Cai et al. 2024 method using t2m and tp.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

path_str = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_' + clim_index +'.nc'
psi_array.to_netcdf(path_str)

sys.exit()






psi_T = xr.DataArray(data = corrs_array_1,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="psi_T, air temperature teleconnection strength inspired by Cai et al. 2024 method with influence of ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index)
                        )
psi_T.to_netcdf('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiT_NINO3_cai_0d5.nc')

psi_P = xr.DataArray(data = corrs_array_2,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="psi_P, air temperature teleconnection strength inspired by Cai et al. 2024 method with influence of ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index)
                        )
psi_P.to_netcdf('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiP_NINO3_cai_0d5.nc')

psi_T_pval = xr.DataArray(data = pvals_array_1,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="p-vals of psi_T, air temperature teleconnection strength inspired by Cai et al. 2024 method with influence of ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index)
                        )
psi_T_pval.to_netcdf('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/pval_psiT_NINO3_cai_0d5.nc')

psi_P_pval = xr.DataArray(data = pvals_array_2,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="p-vals of psi_P, air temperature teleconnection strength inspired by Cai et al. 2024 method with influence of ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index)
                        )
psi_P_pval.to_netcdf('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/pval_psiP_NINO3_cai_0d5.nc')
