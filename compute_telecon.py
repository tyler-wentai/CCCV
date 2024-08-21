import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import rasterio
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime, timedelta
import xarray as xr

print('\n\nSTART ---------------------\n')

file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc'
file_path_POP = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'
file_path_ONI = 'data/NOAA_ONI_data.txt'


# ONI data_frame setup
df_oni = pd.read_csv(file_path_ONI, sep='\s+')
season_to_months = {
    'NDJ': '01',
    'DJF': '02',
    'JFM': '03',
    'FMA': '04',
    'MAM': '05',
    'AMJ': '06',
    'MJJ': '07',
    'JJA': '08',
    'JAS': '09',
    'ASO': '10',
    'SON': '11',
    'OND': '12'
}
df_oni['MN'] = df_oni['SEAS'].map(season_to_months)
df_oni['date'] = pd.to_datetime(df_oni['YR'].astype(str) + '-' + df_oni['MN'] + '-01')
df_oni['date'] = pd.to_datetime(df_oni['date'])
df_oni.set_index('date', inplace=True)
df_oni = df_oni.sort_index()
df_oni = df_oni.drop(columns=['SEAS','YR','MN'])

target_date = datetime(1985, 1, 1, 0, 0, 0)
end_date = datetime(2021, 1, 1, 0, 0, 0)    # end date will be one month AFTER actual desired end date
start_time_ind = int(np.where(df_oni.index == target_date)[0])
end_time_ind = int(np.where(df_oni.index == end_date)[0])

df_oni = df_oni.iloc[start_time_ind:end_time_ind]


# Initialize the air data
dat = nc.Dataset(file_path_AIR)

VAR1=dat.variables['air']

lat = dat.variables['lat'][:]
lon = dat.variables['lon'][:]
time = dat.variables['time'][:]

# Define the reference date: 1800-01-01 00:00:00
reference_date = datetime(1800, 1, 1, 0, 0, 0)

dates = np.array([reference_date + timedelta(hours=int(h)) for h in time])
start_time_ind = int(np.where(dates == target_date)[0])
end_time_ind = int(np.where(dates == end_date)[0])
VAR1 = VAR1[start_time_ind:end_time_ind, :, :]

# Initialize a new array to store the standardized data
VAR1_standard = np.empty_like(VAR1)

# Get the shape of the data array
n_lat, n_long, n_time = VAR1.shape
print(n_lat, n_long, n_time)

# Loop through each (lat, long) point and standardize the time series at each grid point
for i in range(n_lat):
    for j in range(n_long):
        time_series = VAR1[:, i, j]
        mean = np.mean(time_series)
        std = np.std(time_series)
        # Standardize the time series (avoid division by zero)
        if std != 0:
            VAR1_standard[:, i, j] = (time_series - mean) / std
        else:
            VAR1_standard[:, i, j] = time_series  # No standardization if std is zero
            print("std=0!")

# CHECK IF TIME AXIS FOR ONI and VAR1 ARE IDENTICAL
oni_time = df_oni.index.strftime('%Y-%m-%d').to_numpy()
vectorized_format = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))
VAR1_time  = vectorized_format(dates[start_time_ind:end_time_ind])
if not np.array_equal(oni_time, VAR1_time):
        raise ValueError("The two arrays are not identical.")
else:
     print("---The two time arrays are identical.---")
month_start = int(df_oni.index[0].month)


# compute cor(T,ONI)
print(VAR1_standard.shape)
print(VAR1_standard.shape[1])
print(VAR1_standard.shape[2])
rho_tilde = np.empty((12,\
                      VAR1_standard.shape[1],\
                      VAR1_standard.shape[2]))


# psi = np.zeros((VAR1_standard.shape[1],
#                 VAR1_standard.shape[2]))
# lag = 2
# Rval = 3
# psi_array = xr.DataArray(data = psi,
#                          coords={
#                               "lat": lat,
#                               "lon": lon
#                          },
#                          dims = ["lat", "lon"],
#                          attrs=dict(
#                             description="Psi, teleconnection strength via Hsiang 2011 method.",
#                             cor_calc_start_date = str(target_date),
#                             cor_calc_end_date = str(end_date),
#                             L_lag = lag,
#                             R_val = Rval)
#                         )
# psi_array.to_netcdf("/Users/tylerbagwell/Desktop/psi_Hsiang2011_dataset.nc")

# print(psi_array)

# sys.exit()



lag = 2
Rval = 3
alpha_lvl = 0.1
for m in range(12):
    m_num = m+1
    print('month: ', m_num)
    for i in range(n_lat):
        print('...', i)
        for j in range(n_long):
            df_help = pd.DataFrame({
                'month': df_oni.index.month,
                'air_ts': df_oni['ANOM'],
                'oni_ts': VAR1_standard[:, i, j]})
            lag_string = 'oni_ts_lag' + str(lag) + 'm'
            df_help[lag_string] = df_help['oni_ts'].shift((lag))
            df_help = df_help.dropna()
            df_help = df_help[df_help['month'] == m_num]
            pearsonr_result = pearsonr(df_help[lag_string], df_help['air_ts'])
            #if (pearsonr_result[0]>0 and pearsonr_result[1]<alpha_lvl):
            if (pearsonr_result[1]<alpha_lvl):
                 rho_tilde[m,i,j] = 1
            else:
                 rho_tilde[m,i,j] = 0

Mxl = np.sum(rho_tilde, axis=0)
psi = np.where(Mxl >= Rval, 1, 0)
psi_array = xr.DataArray(data = psi,
                         coords={
                              "lat": lat,
                              "lon": lon
                         },
                         dims = ["lat", "lon"],
                         attrs=dict(
                            description="Psi, teleconnection strength via Hsiang 2011 method.",
                            cor_calc_start_date = str(target_date),
                            cor_calc_end_date = str(end_date),
                            L_lag = lag,
                            R_val = Rval)
                        )

print(psi_array)
print("\n", psi_array.values)

psi_array.to_netcdf("/Users/tylerbagwell/Desktop/psi_Hsiang2011_pm.nc")