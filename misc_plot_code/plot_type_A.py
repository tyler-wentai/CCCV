import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime, timedelta
import xarray as xr
import seaborn as sns
import matplotlib.image as mpimg
import geopandas as gpd

print('\n\nSTART ---------------------\n')

file_path_AIR = '/Users/tylerbagwell/Desktop/cccv_data_local/air.2m.mon.mean.nc'
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



i = int(np.where(lat == 0.0)[0]) #80
j = int(np.where(lon == 210.0)[0]) #120
df_help1 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = int(np.where(lat == 60.0)[0]) #354
j = int(np.where(lon == 100.0)[0]) #891
df_help2 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = int(np.where(lat == -3.0)[0]) #639
j = int(np.where(lon == 115.0)[0]) #974
df_help3 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})

i = int(np.where(lat == -30.0)[0]) #431
j = int(np.where(lon == 25.0)[0]) #752
df_help4 = pd.DataFrame({
                'month': df_oni.index.month,
                'oni_ts': df_oni['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})


############### Note: This is a work around since sns.jointplot does not cooperate with plt.subplots

############### 1. CREATE PLOTS
#
g1 = sns.jointplot(data=df_help1, x='oni_ts', y='air_ts', kind="reg")
g1.fig.suptitle("Grid point: (-150.0, 0.0)", y=1.02)
#
g2 = sns.jointplot(data=df_help2, x='oni_ts', y='air_ts', kind="reg")
g2.fig.suptitle("Grid point: (100.0, 60.0)", y=1.02)
#
g3 = sns.jointplot(data=df_help3, x='oni_ts', y='air_ts', kind="reg")
g3.fig.suptitle("Grid point: (115.0, -3.0)", y=1.02)
#
g4 = sns.jointplot(data=df_help4, x='oni_ts', y='air_ts', kind="reg")
g4.fig.suptitle("Grid point: (25.0, -30.0)", y=1.02)

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
# plt.savefig('plots/scatterplots_rho_ONI_airtemp_withpoints', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()




##############################
##############################

# psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/cccv_data_local/rho_airVSoni_lag0.nc')
# psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
# psi = psi.sortby('lon')
# lat = psi['lat'].values
# lon = psi['lon'].values
# variable0 = psi.values[0,:,:]

# path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
# path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
# gdf1 = gpd.read_file(path_land)
# gdf2 = gpd.read_file(path_maritime_0)

# #### PLOTTING
# vmin = np.min(variable0)
# vmax = np.max(variable0)
# print(vmin, vmax)
# levels=np.arange(-0.4,+0.80,0.10) # this sets the colorbar levels
# # colors = ['#ccdbfd', '#e2eafc', '#fff0f3','#ff8fa3', '#ff4d6d', '#a4133c', '#800f2f']
# colors = ['#023e8a', '#0096c7', '#48cae4', '#caf0f8','#ffba08', '#e85d04', "#d00000", "#9d0208", '#6a040f', '#370617', '#03071e', 'k']

# fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# fig.suptitle(r'Correlation $\rho$ of ONI and Air Temp.', fontsize=16)

# lat_points = [0.0, 60.0, -3.0, -30.0]
# lon_points = [-150.0, 100.0, 115.0, 25.0]

# c = ax.contourf(lon, lat, variable0, colors=colors, levels=levels)
# # gdf2.plot(ax=ax, edgecolor=None, color='white')
# gdf1.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
# ax.scatter(lon_points, lat_points, color='lime', marker='X', s=75, edgecolors='k')
# ax.set_title('airtemp_month_lag=0')
# ax.set_xlim([-180.0, 180.0])
# ax.set_ylim([-90.0, +90.0])
# fig.colorbar(c, ax=ax, orientation='horizontal', fraction=0.1, pad=0.1, aspect=30)

# fig.tight_layout()
# plt.savefig('plots/rho_ONI_airtemp_withpoints', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()

