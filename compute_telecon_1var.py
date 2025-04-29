import numpy as np
from scipy.stats import linregress
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
from pingouin import partial_corr
from oldcode.prepare_index import *
from pathlib import Path
import warnings

print('\n\nSTART ---------------------\n')

import xarray as xr

clim_index = 'ANI'

start_year  = 1950
end_year    = 2022 #note that spi only included observations up to 2022

file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/mrsos_ERA5_mon_194001-202212.nc' 
var1str = 'mrsos' # soil moisture



ds1 = xr.open_dataset(file_path_VAR1)

ds1 = ds1.assign_coords(time=ds1.time.dt.floor('D'))


# change dates to time format:
ds1 = ds1.rename({'lat': 'latitude'})
ds1 = ds1.rename({'lon': 'longitude'})

# Access longitude and latitude coordinates
lon1 = ds1['longitude']
lat1 = ds1['latitude']

resolution = 0.5

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
# clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))
clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
                         end_date=datetime(end_year, 12, 1, 0, 0, 0))

common_lon  = np.intersect1d(ds1['longitude'], ds1['longitude'])
common_lat  = np.intersect1d(ds1['latitude'], ds1['latitude'])
common_time = np.intersect1d(ds1['time'], clim_ind.index.to_numpy())

print(common_lon)
print(common_lat)

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
            .apply(_remove_trend))#, include_groups=False))

    # --- 3. standardize 
    sigma = np.array([df.loc[df["month"] == m, "detr"].std(ddof=0) for m in range(12)]) # population sigma

    if israin:
        sigma = np.where(sigma == 0, 1.0, sigma)   # keep dry points at 0

    standardized = df["detr"].values / sigma[months]
    return standardized.tolist()


anom_file1 = Path('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/mrsos_anom_ERA5_0d5_19502022.npy')

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

    np.save('/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/mrsos_anom_ERA5_0d5_' + str(start_year) + str(end_year) + '.npy', var1_std)


# Compute the annualized index value:
clim_ind_common.index = pd.to_datetime(clim_ind_common.index)     # Ensure 'date' to datetime and extract year & month
clim_ind_common = clim_ind_common.copy()
clim_ind_common['year'] = clim_ind_common.index.year
clim_ind_common['month'] = clim_ind_common.index.month

## --- NINO3
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

# print(index_AVG)

## --- ANI
jun_jul_aug_df = clim_ind_common[clim_ind_common['month'].isin([6, 7, 8])].copy() # prepare June, July, August (JJA) data for current year
jun     = jun_jul_aug_df[jun_jul_aug_df['month'] == 6][['year', 'ANOM']].rename(columns={'ANOM': 'JUN_ANOM'})
jul     = jun_jul_aug_df[jun_jul_aug_df['month'] == 7][['year', 'ANOM']].rename(columns={'ANOM': 'JUL_ANOM'})
aug     = jun_jul_aug_df[jun_jul_aug_df['month'] == 8][['year', 'ANOM']].rename(columns={'ANOM': 'AUG_ANOM'})

yearly = pd.merge(jun, jul, on='year', how='inner') # merge June, July, August data
yearly = pd.merge(yearly, aug, on='year', how='inner') # merge June, July, August data

yearly['avg_ANOM'] = yearly[['JUN_ANOM', 'JUL_ANOM', 'AUG_ANOM']].mean(axis=1) # Calculate the average JJA ANOM value
index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

print(index_AVG)

####
# n_months = 12 # NINO3
# n_months = 8 # DMI
n_months = 8 # ANI 

corrs_array_1   = np.empty((n_months,n_lat,n_long))
pvals_array_1   = np.empty((n_months,n_lat,n_long))
monthly_psi     = np.empty((n_months,n_lat,n_long))
psi             = np.empty((n_lat,n_long))

# index_dat['avg_ANOM_ENSO'] = index_dat['avg_ANOM_ENSO'].shift(-1) # NEED TO TEST WHICH YEAR OF ENSO DJF TO CORRELATE!!!!
# index_dat = index_dat.dropna(subset=['avg_ANOM_ENSO'])

print("\nComputing psi array...")
#warnings.filterwarnings('ignore', category=RuntimeWarning) #!!!!!!!
for i in range(n_lat):
    if (i%10==0): 
        print("...", i)
    for j in range(n_long):
        current_vars = pd.DataFrame(data=var1_std[:,i,j],
                                            index=var1_common['time'], #need to use var1_common since it still contains the time data
                                            columns=[var1str])
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
        # for k in range(1,int(n_months+1),1):
        #     # may-dec of year t
        #     var_ts = current_vars[current_vars['month'] == int(k+4)].copy()

        ### ANI
        for k in range(1,int(n_months+1),1):
            # may-dec of year t
            var_ts = current_vars[current_vars['month'] == int(k+4)].copy()

            # compute correlations of yearly month, k, air anomaly with index 
            var_ts = pd.merge(var_ts, index_AVG, how='inner', on='year')

            has_nan = var_ts[[var1str, 'avg_ANOM']].isna().any().any()
            if not has_nan:
                corr_1 = partial_corr(data=var_ts, x='avg_ANOM', y=var1str)

                corrs_array_1[int(k-1),i,j] = corr_1['r'].values[0]
                pvals_array_1[int(k-1),i,j] = corr_1['p-val'].values[0]

            else:
                corrs_array_1[int(k-1),i,j] = np.nan
                pvals_array_1[int(k-1),i,j] = 1.

            # save monthly psi values
            var1_psi = corr_1['r'].iloc[0] if corr_1['p-val'].iloc[0] < 0.05 else 0
            monthly_psi[int(k-1),i,j] = var1_psi

        corrs1 = pd.Series(corrs_array_1[:,i,j])
        pvals1 = pd.Series(pvals_array_1[:,i,j])

        has_nan = corrs1.isna().any()
        if has_nan==False:
            # var1
            corr1_total_sum = corrs1[pvals1 < 0.05].sum()
            # compute teleconnection (psi)
            psi[i,j] = corr1_total_sum
        else:
            psi[i,j] = np.nan

psi_array = xr.DataArray(data = psi,
                            coords={
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["lat", "lon"],
                        attrs=dict(
                            description="Teleconnection strength (Psi) inspired by Cai et al. 2024 method using ERA5 spi6.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathA_str = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_' + var1str + clim_index +'.nc'
psi_array.to_netcdf(pathA_str)

psiMonthly_array = xr.DataArray(data = monthly_psi,
                            coords={
                            "month": np.arange(1,(n_months+1),1),    
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["month", "lat", "lon"],
                        attrs=dict(
                            description="Monthly teleconnection strength (Psi_m) inspired by Cai et al. 2024 method using ERA5 spi6.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathB_str = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiMonthly_' + var1str + clim_index +'.nc'
psiMonthly_array.to_netcdf(pathB_str)

sys.exit()