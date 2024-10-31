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

print('\n\nSTART ---------------------\n')


#
def prepare_NINO3(file_path, start_date, end_date):
    """
    Prepare NINO3 index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nino3 = pd.read_csv(file_path, sep=r'\s+', skiprows=1, skipfooter=7, header=None, engine='python')
    year_start = int(nino3.iloc[0,0])
    nino3 = nino3.iloc[:,1:nino3.shape[1]].values.flatten()
    df_nino3 = pd.DataFrame(nino3)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_nino3.shape[0], freq='MS')
    df_nino3.index = date_range
    df_nino3.rename_axis('date', inplace=True)
    df_nino3.columns = ['ANOM']

    start_ts_l = np.where(df_nino3.index == start_date)[0]
    end_ts_l = np.where(df_nino3.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of NINO3 index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of NINO3 index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_nino3 = df_nino3.iloc[start_ts_ind:end_ts_ind]

    return df_nino3


#
def prepare_DMI(file_path, start_date, end_date):
    """
    Prepare DMI index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/data/timeseries/monthly/standard.html)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    dmi = pd.read_csv(file_path, sep=r'\s+', skiprows=1, skipfooter=7, header=None, engine='python')
    year_start = int(dmi.iloc[0,0])
    dmi = dmi.iloc[:,1:dmi.shape[1]].values.flatten()
    df_dmi = pd.DataFrame(dmi)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_dmi.shape[0], freq='MS')
    df_dmi.index = date_range
    df_dmi.rename_axis('date', inplace=True)
    df_dmi.columns = ['ANOM']

    start_ts_l = np.where(df_dmi.index == start_date)[0]
    end_ts_l = np.where(df_dmi.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of DMI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of DMI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_dmi = df_dmi.iloc[start_ts_ind:end_ts_ind]

    return df_dmi

####

# file_path_VAR1 = '/Users/tylerbagwell/Desktop/spei6_ERA5_mon_194001-202212.nc' # SPEI

# ds1 = xr.open_dataset(file_path_VAR1)
# var1str = 'spei6'
# var1 = ds1[var1str]

# cutoff_date = '1980-01-01'  # Replace with your desired cutoff date
# var1_subset = var1.sel(time=slice(cutoff_date, None))

# mask = var1_subset.notnull().all(dim='time')

# # Extract 1D latitude and longitude arrays
# lat = var1_subset['lat'].values    # Shape: (lat,)
# lon = var1_subset['lon'].values    # Shape: (lon,)

# # Create 2D grids of latitude and longitude
# # The resulting lat2d and lon2d have shape (lat, lon)
# lon2d, lat2d = np.meshgrid(lon, lat)

# # Apply the mask to flatten the 2D grids and filter valid points
# valid_lats = lat2d[mask.values]  # 1D array of valid latitudes
# valid_lons = lon2d[mask.values]  # 1D array of valid longitudes





# plt.figure(figsize=(15, 8))
# import cartopy.crs as ccrs
# # ax = plt.axes(projection=ccrs.PlateCarree())

# # # Add map features
# # ax.coastlines()
# # ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

# # Scatter plot of valid (lat, lon) points
# plt.scatter(
#     valid_lons, 
#     valid_lats, 
#     s=0.01,               # Marker size
#     color='blue',       # Marker color
#     alpha=0.6,          # Transparency
#     label='Valid spei6 Points'
# )

# # Add a title and legend
# plt.title('Latitude and Longitude Points with No NaN in spei6')
# plt.legend()

# # Show the plot
# plt.show()


# sys.exit()
###


start_year  = 1980
end_year    = 2022

file_path_VAR1 = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc' # SPEI

import xarray as xr

ds1 = xr.open_dataset(file_path_VAR1)
var1str = 'air'
var1 = ds1[var1str]  # DataArray from the first dataset

# Access longitude and latitude coordinates
lon1 = ds1['lon']
lat1 = ds1['lat']


lat_int_mask = (lat1 % 0.5 == 0)
lon_int_mask = (lon1 % 0.5 == 0)

ds_cleaned = ds1.sel(lat=lat1[lat_int_mask], lon=lon1[lon_int_mask])

print(ds_cleaned['lat'].values)
print(ds_cleaned['lon'].values)

ds1 = ds_cleaned.copy()

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    lon = ds['lon']
    lon = ((lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon)
    return ds

# Apply conversion if necessary
if lon1.max() > 180:
    ds1 = convert_longitude(ds1)

ds1 = ds1.sortby('lon')
lon1 = ds1['lon']
lat1 = ds1['lat']

ds1 = ds1.assign_coords(
    lon=np.round(ds1['lon'], decimals=2),
    lat=np.round(ds1['lat'], decimals=2)
    )


# load index data
clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
                         end_date=datetime(end_year, 12, 1, 0, 0, 0))

# clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
#                          start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                          end_date=datetime(end_year, 12, 1, 0, 0, 0))

common_time = np.intersect1d(ds1['time'], clim_ind.index.to_numpy())


ds1_common      = ds1.sel(time=common_time)
var1_common = ds1_common[var1str]

clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]


# Check shapes
print("var1_common shape:", var1_common.shape)
print("clim_ind shape:   ", clim_ind_common.shape)

n_time, n_lat, n_long = var1_common.shape


# Verify that coordinates are identical
assert np.array_equal(var1_common['time'], clim_ind_common.index)

def standardize_and_detrend_monthly(data):
    data = np.array(data)
    n = len(data)
    months = np.arange(n) % 12  # Assign month indices 0-11
    means = np.array([data[months == m].mean() for m in range(12)])
    stds = np.array([data[months == m].std() for m in range(12)])

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

var1_std = np.empty_like(var1_common) # Initialize a new array to store the standardized data

# var1_std = var1_common # DELETE LATER!!!!!!!!!
# var2_std = var2_common # DELETE LATER!!!!!!!!!


print("Standardizing climate variable data...")
for i in range(n_lat):
    if (i%10==0): 
        print("...", i)
    for j in range(n_long):
        has_nan = np.isnan(var1_common[:, i, j]).any()
        if (has_nan==False):
            var1_std[:, i, j] = standardize_and_detrend_monthly(var1_common[:, i, j])
        else: 
            var1_std[:, i, j] = var1_common[:, i, j]


# Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
# For pIOD, we use SEP(t),OCT(t),NOV(t)
clim_ind_common.index = pd.to_datetime(clim_ind_common.index)     # Ensure 'date' to datetime and extract year & month
clim_ind_common['year'] = clim_ind_common.index.year
clim_ind_common['month'] = clim_ind_common.index.month


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



# sep_oct_nov_df = clim_ind_common[clim_ind_common['month'].isin([9, 10, 11])].copy() # prepare January and February data for current year
# sep     = sep_oct_nov_df[sep_oct_nov_df['month'] == 9][['year', 'ANOM']].rename(columns={'ANOM': 'SEP_ANOM'})
# oct     = sep_oct_nov_df[sep_oct_nov_df['month'] == 10][['year', 'ANOM']].rename(columns={'ANOM': 'OCT_ANOM'})
# nov     = sep_oct_nov_df[sep_oct_nov_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})

# yearly = pd.merge(sep, oct, on='year', how='inner') # merge December, January, and February data
# yearly = pd.merge(yearly, nov, on='year', how='inner') # merge December, January, and February data

# yearly['avg_ANOM'] = yearly[['SEP_ANOM', 'OCT_ANOM', 'NOV_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
# index_AVG = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)




# ENSO: Compute monthly correlation and teleconnection (psi) at each grid point, computes correlations for each month from JUN(t-1) to AUG(t) with DJF index(t)
# IOD: Compute monthly correlation and teleconnection (psi) at each grid point, computes correlations for each month from MAY(t) to MAY(t+1) with SON index(t)
corrs_array_1 = np.empty((15,n_lat,n_long))
psi = np.empty((n_lat,n_long))

print("\nComputing psi array...")
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
        for k in range(1,16,1):
            # jun-dec of year t-1
            if (k<=7):
                var_ts = current_vars[current_vars['month'] == int(k+6-1)].copy()
                var_ts['year'] = var_ts['year'] + 1  # Shift to next year
            # jan-aug of year t
            else:
                var_ts = current_vars[current_vars['month'] == int(k-7)].copy()

            # compute correlations of yearly month, k, air anomaly with index 
            var_ts = pd.merge(var_ts, index_AVG, how='inner', on='year')
    
            has_nan = var_ts[var1str].isna().any()
            if has_nan==False:
                partial_corr_1 = partial_corr(data=var_ts, x=var1str, y='avg_ANOM')['r'].values[0]
                corrs_array_1[int(k-1),i,j] = partial_corr_1
            else:
                corrs_array_1[int(k-1),i,j] = np.nan

        corrs1 = pd.Series(corrs_array_1[:,i,j])

        has_nan = corrs1.isna().any()
        if has_nan==False:
            # var1
            rolling_avg1 = corrs1.rolling(window=3, center=False).mean() ### BE AWARE OF CENTERING OF WINDOW!!!
            #rolling_avg1 = np.abs(rolling_avg1)
            # max_corr1 = np.nanmax(rolling_avg1)
            index = np.argmax(np.abs(rolling_avg1))
            max_corr1 = rolling_avg1[index] # find the extrema value (either neg or pos)
            # compute teleconnection (psi)
            psi[i,j] = max_corr1
        else:
            psi[i,j] = np.nan

psi_array = xr.DataArray(data = psi,
                            coords={
                            "lat": lat1,
                            "lon": lon1
                        },
                        dims = ["lat", "lon"],
                        attrs=dict(
                            description="Psi, teleconnection strength (w/ sign) inspired by Callahan 2023 method using air.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = 'NINO3')
                        )

psi_array.to_netcdf('/Users/tylerbagwell/Desktop/psi_callahan_nino3_air_pm_0d5deg.nc') 