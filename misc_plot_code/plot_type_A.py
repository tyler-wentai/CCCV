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
import seaborn as sns
import matplotlib.image as mpimg

print('\n\nSTART ---------------------\n')

file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc'
#file_path_POP = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'
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

target_date = datetime(1980, 1, 1, 0, 0, 0)
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
n_time, n_lat, n_long = VAR1.shape
print(n_time, n_lat, n_long)

# Loop through each (lat, long) point and standardize the time series at each grid point
for i in range(n_lat):
    print(i)
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


i = 80
j = 120
df_help1 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = 354
j = 891
df_help2 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = 639
j = 974
df_help3 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = 431
j = 752
df_help4 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})


############### Note: This is a work around since sns.jointplot does not cooperate with plt.subplots

############### 1. CREATE PLOTS
g1 = sns.jointplot(data=df_help1, x='oni_ts', y='air_ts', kind="reg")
g2 = sns.jointplot(data=df_help2, x='oni_ts', y='air_ts', kind="reg")
g3 = sns.jointplot(data=df_help3, x='oni_ts', y='air_ts', kind="reg")
g4 = sns.jointplot(data=df_help4, x='oni_ts', y='air_ts', kind="reg")

############### 2. SAVE PLOTS IN MEMORY TEMPORALLY
g1.savefig('g1.png')
plt.close(g1.fig)

g2.savefig('g2.png')
plt.close(g2.fig)

g3.savefig('g3.png')
plt.close(g3.fig)

g4.savefig('g4.png')
plt.close(g4.fig)

############### 3. CREATE YOUR SUBPLOTS FROM TEMPORAL IMAGES
f, axarr = plt.subplots(2, 2, figsize=(10, 10))

axarr[0,0].imshow(mpimg.imread('g1.png'))
axarr[0,1].imshow(mpimg.imread('g2.png'))
axarr[1,0].imshow(mpimg.imread('g3.png'))
axarr[1,1].imshow(mpimg.imread('g4.png'))

# turn off x and y axis
[ax.set_axis_off() for ax in axarr.ravel()]

plt.tight_layout()
plt.show()


