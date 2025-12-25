import numpy as np
from scipy.stats import linregress
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
from pingouin import partial_corr
from utils.calc_annual_index import *
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view

print('\n\nSTART ---------------------\n')
# COMPUTES THE TELECONNECTION STRENGTH (PSI) USING THE CALLAHAN AND MANKIN 2023 METHOD

clim_index = 'DMI'

start_year  = 1950
end_year    = 2023

file_path_VAR1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/raw_climate_data/ERA5_t2m_raw.nc' # air temperature 2 meter
file_path_VAR2 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/raw_climate_data/ERA5_tp_raw.nc'  # total precipitation


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
# clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Eindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))

enso_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
                        end_date=datetime(end_year+1, 12, 1, 0, 0, 0))

common_lon  = np.intersect1d(ds1['longitude'], ds2['longitude']) #probably should check that this is not null
common_lat  = np.intersect1d(ds1['latitude'], ds2['latitude'])
common_time = np.intersect1d(ds1['time'], ds2['time'])
common_time = np.intersect1d(common_time, clim_ind.index.to_numpy())

ds1_common      = ds1.sel(time=common_time, longitude=common_lon, latitude=common_lat)
ds2_common      = ds2.sel(time=common_time, longitude=common_lon, latitude=common_lat)
clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]

var1_common = ds1_common[var1str]
var2_common = ds2_common[var2str]

# need to extend common_time_enso by one month to account for Jan of following year
ct = pd.DatetimeIndex(common_time)
ct = ct.append(pd.DatetimeIndex([ct[-1] + pd.offsets.MonthBegin(1)]))
common_time_enso = ct
enso_ind_common = enso_ind.loc[enso_ind.index.isin(pd.to_datetime(common_time_enso))]

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
# assert np.array_equal(clim_ind_common.index, enso_ind_common.index)


def detrend_then_standardize_monthly(data, israin: bool = False):
    """
    1. Remove the mean seasonal cycle (12-month climatology)
    2. Detrend each calendar-month slice separately
    3. Divide by the post-detrend sigma for that month
    """
    # set‑up
    data   = np.asarray(data, dtype=float)
    n      = data.size
    months = np.arange(n) % 12            # 0...11

    # 1. climatology 
    clim_mean = np.array([data[months == m].mean() for m in range(12)])
    anom      = data - clim_mean[months]  # mean removed

    # 2. detrend each calendar month 
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

    # 3. standardize 
    sigma = np.array([df.loc[df["month"] == m, "detr"].std(ddof=0) for m in range(12)]) # population sigma

    if israin:
        sigma = np.where(sigma == 0, 1.0, sigma)   # keep dry points at 0

    standardized = df["detr"].values / sigma[months]
    return standardized.tolist()


anom_file1 = Path('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_climate_data/t2m_anom_ERA5_0d5_19502023_FINAL.npy')
anom_file2 = Path('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_climate_data/tp_anom_ERA5_0d5_19502023_FINAL.npy')

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

    print("Standardizing and de-trending climate variable data...")
    for i in range(n_lat):
        if (i%10==0): 
            print("...", i)
        for j in range(n_long):
            var1_std[:, i, j] = detrend_then_standardize_monthly(var1_common[:, i, j])
            has_nan = np.isnan(var2_common[:, i, j]).any()
            if (has_nan==False):
                var2_std[:, i, j] = detrend_then_standardize_monthly(var2_common[:, i, j], israin=True)
            else: 
                var2_std[:, i, j] = var2_common[:, i, j]

    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/t2m_anom_ERA5_0d5_' + str(start_year) + str(end_year) + '_FINAL.npy', var1_std)
    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/tp_anom_ERA5_0d5_' + str(start_year) + str(end_year) + '_FINAL.npy', var2_std)


# var2_monthly_array = xr.DataArray(data = var2_std,
#                             coords={
#                             "time": common_time,    
#                             "lat": common_lat,
#                             "lon": common_lon
#                         },
#                         dims = ["time", "lat", "lon"],
#                         attrs=dict(
#                             description="Gridded monthly ERA5 tp anomaly data.",
#                             psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
#                             psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
#                             climate_index_used = clim_index,
#                             resolution = resolution)
#                         )

# path2_str = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/tp_anom_ERA5_0d5_19502023_wTimeLatLon.nc'
# var2_monthly_array.to_netcdf(path2_str)


# Compute the annualized index value:
clim_ind_common.index = pd.to_datetime(clim_ind_common.index)     # Ensure 'date' to datetime and extract year & month
clim_ind_common = clim_ind_common.copy()
clim_ind_common['year'] = clim_ind_common.index.year
clim_ind_common['month'] = clim_ind_common.index.month

enso_ind_common.index = pd.to_datetime(enso_ind_common.index)     # Ensure 'date' to datetime and extract year & month
enso_ind_common = enso_ind_common.copy()
enso_ind_common['year'] = enso_ind_common.index.year
enso_ind_common['month'] = enso_ind_common.index.month

# --- ENSO
jan_df = enso_ind_common[enso_ind_common['month'] == 1].copy() # prepare January data from following year
jan_df['year'] = jan_df['year'] - 1  # Shift back a year
jan_df = jan_df[['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})

nov_dec_df = enso_ind_common[enso_ind_common['month'].isin([11, 12])].copy() # prepare November and December data for current year
nov     = nov_dec_df[nov_dec_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})
dec     = nov_dec_df[nov_dec_df['month'] == 12][['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

yearly = pd.merge(jan_df, nov, on='year', how='inner') # merge November_t, January_t+1 data
yearly = pd.merge(yearly, dec, on='year', how='inner') # merge November_t, December_t, January_t+1 data

yearly['avg_ENSO'] = yearly[['NOV_ANOM', 'DEC_ANOM', 'JAN_ANOM']].mean(axis=1) # Calculate the average NDJ ANOM value
enso_AVG = yearly[['year', 'avg_ENSO']].sort_values('year').reset_index(drop=True)

print(enso_AVG)

## --- NINO3, NINO34
# jan_df = clim_ind_common[clim_ind_common['month'] == 1].copy() # prepare January data from following year
# jan_df['year'] = jan_df['year'] - 1  # Shift back a year
# jan_df = jan_df[['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})

# nov_dec_df = clim_ind_common[clim_ind_common['month'].isin([11, 12])].copy() # prepare November and December data for current year
# nov     = nov_dec_df[nov_dec_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})
# dec     = nov_dec_df[nov_dec_df['month'] == 12][['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

# yearly = pd.merge(jan_df, nov, on='year', how='inner') # merge November_t, January_t+1 data
# yearly = pd.merge(yearly, dec, on='year', how='inner') # merge November_t, December_t, January_t+1 data

# yearly['avg_ANOM'] = yearly[['NOV_ANOM', 'DEC_ANOM', 'JAN_ANOM']].mean(axis=1) # Calculate the average NDJ ANOM value
# index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

# print(index_AVG)

# may_to_dec_df = ENSO_ind_common[ENSO_ind_common['month'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])].copy() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
# index_AVG = may_to_dec_df.groupby('year')['ANOM'].mean().reset_index() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
# index_AVG = index_AVG.rename(columns={'ANOM': 'avg_ANOM'}) # DELETE !!!!!!!!!!!!!!!!!!!!!!!!

## --- DMI
sep_oct_nov_df = clim_ind_common[clim_ind_common['month'].isin([9, 10, 11])].copy() # prepare January and February data for current year
sep     = sep_oct_nov_df[sep_oct_nov_df['month'] == 9][['year', 'ANOM']].rename(columns={'ANOM': 'SEP_ANOM'})
oct     = sep_oct_nov_df[sep_oct_nov_df['month'] == 10][['year', 'ANOM']].rename(columns={'ANOM': 'OCT_ANOM'})
nov     = sep_oct_nov_df[sep_oct_nov_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})

yearly = pd.merge(sep, oct, on='year', how='inner') # merge December, January, and February data
yearly = pd.merge(yearly, nov, on='year', how='inner') # merge December, January, and February data

yearly['avg_ANOM'] = yearly[['SEP_ANOM', 'OCT_ANOM', 'NOV_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

print(index_AVG)


####
# n_months = 12 # NINO3
n_months = 8 # DMI

corrs_array_1   = np.empty((n_months,n_lat,n_long))
pvals_array_1   = np.empty((n_months,n_lat,n_long))
corrs_array_2   = np.empty((n_months,n_lat,n_long))
pvals_array_2   = np.empty((n_months,n_lat,n_long))
monthly_psi     = np.empty((n_months,n_lat,n_long))
psi             = np.empty((n_lat,n_long))

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
        ### NINO3 / NINO34
        # for k in range(1,int(n_months+1),1):
        #     # may-dec of year t
        #     if (k<=8):
        #         var_ts = current_vars[current_vars['month'] == int(k+4)].copy()
        #     else:
        #         var_ts = current_vars[current_vars['month'] == int(k-8)].copy()
        #         var_ts['year'] = var_ts['year'] - 1  # Shift to previous year

        ### DMI
        for k in range(1,int(n_months+1),1):
            # may-dec of year t
            var_ts = current_vars[current_vars['month'] == int(k+4)].copy()

            ############
            # compute correlations of yearly month, k, air anomaly with index 
            var_ts = (
                var_ts
                .merge(index_AVG, how="inner", on="year")
                .merge(enso_AVG,  how="inner", on="year")
            )
            has_nan = var_ts[[var1str, var2str, 'avg_ANOM', 'avg_ENSO']].isna().any().any()
            if not has_nan:
                corr_1 = partial_corr(data=var_ts, x='avg_ANOM', y=var1str, covar='avg_ENSO')    # partial corr
                corr_2 = partial_corr(data=var_ts, x='avg_ANOM', y=var2str, covar='avg_ENSO')    # partial corr

                corrs_array_1[int(k-1),i,j] = corr_1['r'].values[0]
                corrs_array_2[int(k-1),i,j] = corr_2['r'].values[0]
                pvals_array_1[int(k-1),i,j] = corr_1['p-val'].values[0]
                pvals_array_2[int(k-1),i,j] = corr_2['p-val'].values[0]

            else:
                corrs_array_1[int(k-1),i,j] = np.nan
                corrs_array_2[int(k-1),i,j] = np.nan
                pvals_array_1[int(k-1),i,j] = 1.
                pvals_array_2[int(k-1),i,j] = 1.

            # save monthly psi values
            var1_psi = corr_1['r'].iloc[0] if corr_1['p-val'].iloc[0] < 0.05 else 0     # not absolute values
            var2_psi = corr_2['r'].iloc[0] if corr_2['p-val'].iloc[0] < 0.05 else 0     # not absolute values
            monthly_psi[int(k-1),i,j] = var1_psi + var2_psi

        corrs1 = pd.Series(corrs_array_1[:,i,j])
        corrs2 = pd.Series(corrs_array_2[:,i,j])
        pvals1 = pd.Series(pvals_array_1[:,i,j])
        pvals2 = pd.Series(pvals_array_2[:,i,j])

        has_nan = corrs1.isna().any()
        if has_nan==False:
            # 
            corrs1[pvals1 > 0.05] = 0
            psi1_monthly = np.abs(corrs1)
            windows1 = sliding_window_view(psi1_monthly, window_shape=3) # 3 month rolling window
            psi1_rolling_avg = windows1.mean(axis=1)
            psi1 = np.max(psi1_rolling_avg)

            corrs2[pvals2 > 0.05] = 0
            psi2_monthly = np.abs(corrs2)
            windows2 = sliding_window_view(psi2_monthly, window_shape=3) # 3 month rolling window
            psi2_rolling_avg = windows2.mean(axis=1)
            psi2 = np.max(psi2_rolling_avg)

            psi[i,j] = psi1 + psi2

        else:
            psi[i,j] = np.nan

psi_array = xr.DataArray(data = psi,
                            coords={
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["lat", "lon"],
                        attrs=dict(
                            description="Teleconnection strength (Psi) using partial correlations inspired by Callahan and Mankin 2023 method using ERA5 t2m and tp, ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathA_str = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_' + clim_index +'_type2_ensoremoved.nc'
psi_array.to_netcdf(pathA_str)

psiMonthly_array = xr.DataArray(data = monthly_psi,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="Monthly raw teleconnection strength (Psi_m) using partial correlations inspired by Callahan and Mankin 2023 method using ERA5 t2m and tp, ENSO removed.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathB_str = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psiMonthly_' + clim_index +'_type2_ensoremoved.nc'
psiMonthly_array.to_netcdf(pathB_str)


