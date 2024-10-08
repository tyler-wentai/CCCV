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
def prepare_ONI(file_path, start_date, end_date):
    """
    Prepare ONI index data as pd.Data.Frame
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    df_oni = pd.read_csv(file_path, sep='\s+')
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

    start_ts_l = np.where(df_oni.index == start_date)[0]
    end_ts_l = np.where(df_oni.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of ONI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of ONI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_oni = df_oni.iloc[start_ts_ind:end_ts_ind]

    return df_oni


#
def prepare_NINO3(file_path, start_date, end_date):
    """
    Prepare NINO3 index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nino3 = pd.read_csv(file_path, sep='\s+', skiprows=1, skipfooter=7, header=None, engine='python')
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
    dmi = pd.read_csv(file_path, sep='\s+', skiprows=1, skipfooter=7, header=None, engine='python')
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


#
def prepare_AMM(file_path, start_date, end_date):
    """
    Prepare AMM (Atlantic Meridional Mode) index data as pd.Data.Frame from Standard PSL Format 
    (https://psl.noaa.gov/data/timeseries/monthly/standard.html)
    Data source: https://psl.noaa.gov/data/timeseries/monthly/AMM/
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    amm = pd.read_csv(file_path, sep='\s+', skiprows=1, skipfooter=5, header=None, engine='python')
    year_start = int(amm.iloc[0,0])
    amm = amm.iloc[:,1:amm.shape[1]].values.flatten()
    df_amm = pd.DataFrame(amm)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_amm.shape[0], freq='MS')
    df_amm.index = date_range
    df_amm.rename_axis('date', inplace=True)
    df_amm.columns = ['ANOM']

    start_ts_l = np.where(df_amm.index == start_date)[0]
    end_ts_l = np.where(df_amm.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of DMI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of DMI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_amm = df_amm.iloc[start_ts_ind:end_ts_ind]

    return df_amm


#
def compute_psi_Hsiang2011(climate_index, start_year, end_year, num_lag, num_R, save_path):
    """
    Computes teleconnection strength (psi) between specified climate_index and air temperature 
    anomaly at all global grid points based on the method of Hsiang 2011 (w/o population weighting).
    Output is a NetCDF file with description.
    """

    # Check if arguments have appropriate type:
    if not isinstance(start_year, int):
        raise TypeError(f"Expected an integer for start_year, but got {type(start_year).__name__}.")
    if not isinstance(end_year, int):
        raise TypeError(f"Expected an integer for end_year, but got {type(end_year).__name__}.")
    if not isinstance(num_lag, int):
        raise TypeError(f"Expected an integer for num_lag, but got {type(num_lag).__name__}.")
    if not isinstance(num_R, int):
        raise TypeError(f"Expected an integer for num_R, but got {type(num_R).__name__}.")

    # Read in data files
    file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc' # Air temperature anomaly
    file_path_ONI = 'data/NOAA_ONI_data.txt' # ONI: Oceanic Nino Index
    file_path_DMI = 'data/NOAA_DMI_data.txt' # DMI: Dipole Mode Index
    file_path_AMM = 'data/NOAA_AMM_data.txt' # AMM: Atlantic Meridional Mode Index

    start_date = datetime(start_year, 1, 1, 0, 0, 0)
    end_date = datetime(end_year, 12, 1, 0, 0, 0)

    # Read in climate index data file specified by climate_index
    if climate_index=="oni" or climate_index=="ONI":
        df_climate_index = prepare_ONI(file_path_ONI, start_date, end_date)
        climate_index_name = 'oni'
    elif climate_index=="dmi" or climate_index=="DMI":
        df_climate_index = prepare_DMI(file_path_DMI, start_date, end_date)
        climate_index_name = 'dmi'
    elif climate_index=="amm" or climate_index=="AMM":
        df_climate_index = prepare_AMM(file_path_AMM, start_date, end_date)
        climate_index_name = 'amm'
    else:
        raise ValueError("Specified climate_index is not a valid climate index name.")
    
    # Read in and initialize the air temperature anomaly data
    dat = nc.Dataset(file_path_AIR)

    VAR1=dat.variables['air']
    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    time = dat.variables['time'][:]

    # Define the reference date: 1800-01-01 00:00:00. This is reference specified by NOAA's air.2m.mon.mean.nc file. 
    reference_date = datetime(1800, 1, 1, 0, 0, 0)

    dates = np.array([reference_date + timedelta(hours=int(h)) for h in time])
    start_time_ind = int(np.where(dates == start_date)[0][0])
    end_time_ind = int(int(np.where(dates == end_date)[0][0]) + 1)
    VAR1 = VAR1[start_time_ind:end_time_ind, :, :]

    VAR1_standard = np.empty_like(VAR1) # Initialize a new array to store the standardized data
    n_time, n_lat, n_long = VAR1.shape
    print("Original shape of air temp. data: ", n_time, n_lat, n_long)

    # Loop through each (lat, long) point and standardize the time series at each grid point
    print("Standardizing air temp. data...")
    for i in range(n_lat):
        if (i%100==0): 
            print("...", i)
        for j in range(n_long):
            time_series = VAR1[:, i, j]
            mean = np.mean(time_series)
            std = np.std(time_series)
            # Standardize the time series (avoid division by zero)
            if std != 0:
                VAR1_standard[:, i, j] = (time_series - mean) / std
            else:
                raise ValueError("std=0 at grid point: (", i, ",", j,")!")
            
    # Check if df_climate_index's and VAR1_standard's ts indicies are idential.
    ind_time = df_climate_index.index.strftime('%Y-%m-%d').to_numpy()
    vectorized_format = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))
    VAR1_time  = vectorized_format(dates[start_time_ind:end_time_ind])
    if not np.array_equal(ind_time, VAR1_time):
        raise ValueError("---The two date arrays of ind_time and VAR1_time (air temp.) are NOT identical.---")
    else:
        print("---The two date arrays of ind_time and VAR1_time (air temp.) are identical.---")
    month_start = int(df_climate_index.index[0].month)

    # Compute correlation(air temp, climate index) at each grid point
    rho_tilde = np.empty((12,\
                          VAR1_standard.shape[1],\
                          VAR1_standard.shape[2]))
    alpha_lvl = 0.1
    for m in range(12):
        m_num = m+1
        print('month: ', m_num)
        for i in range(n_lat):
            if (i%100==0): 
                print("...", i)
            for j in range(n_long):
                df_help = pd.DataFrame({
                    'month': df_climate_index.index.month,
                    'ind_ts': df_climate_index['ANOM'],
                    'air_ts': VAR1_standard[:, i, j]})
                lag_string = 'ind_ts_lag' + str(num_lag) + 'm'
                df_help[lag_string] = df_help['ind_ts'].shift((num_lag)) # created a shifted column shifted by num_lag monthly lags
                df_help = df_help.dropna()
                df_help = df_help[df_help['month'] == m_num]
                pearsonr_result = pearsonr(df_help[lag_string], df_help['air_ts'])
                if (pearsonr_result[0]>0 and pearsonr_result[1]<alpha_lvl): # 1 if correlation is positive and significant at alpha_lvl
                    rho_tilde[m,i,j] = 1
                else:
                    rho_tilde[m,i,j] = 0

    # Sum number of months have rho_tilde=1 for each grid point
    Mxl = np.sum(rho_tilde, axis=0)

    # Define binary "teleconnection strength" of grid point: 
    #   teleconnect when Mxl >= num_R
    #   weakly-affected when Mxl < num_R
    psi = np.where(Mxl >= num_R, 1, 0)

    # Store as xarray.DataArray with informational attributes
    psi_array = xr.DataArray(data = psi,
                             coords={
                              "lat": lat,
                              "lon": lon
                            },
                            dims = ["lat", "lon"],
                            attrs=dict(
                                description="Psi, teleconnection strength via Hsiang 2011 method.",
                                cor_calc_start_date = str(start_date),
                                cor_calc_end_date = str(end_date),
                                climate_index_used = climate_index_name,
                                L_lag = num_lag,
                                R_val = num_R)
                            )

    # Save psi_array as NetCDF file
    psi_array.to_netcdf(save_path)


#
def compute_gridded_correlation(climate_index, start_year, end_year, num_lag, save_path):
    """
    Computes Pearson correlation between climate_index and air temperature anomaly at 
    all global grid points.
    Output is a NetCDF file with description.
    """

    # Check if arguments have appropriate type:
    if not isinstance(start_year, int):
        raise TypeError(f"Expected an integer for start_year, but got {type(start_year).__name__}.")
    if not isinstance(end_year, int):
        raise TypeError(f"Expected an integer for end_year, but got {type(end_year).__name__}.")
    if not isinstance(num_lag, int):
        raise TypeError(f"Expected an integer for num_lag, but got {type(num_lag).__name__}.")

    # Read in data files
    file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc' # Air temperature anomaly
    file_path_ONI = 'data/NOAA_ONI_data.txt' # ONI: Oceanic Nino Index
    file_path_DMI = 'data/NOAA_DMI_data.txt' # DMI: Dipole Mode Index
    file_path_AMM = 'data/NOAA_AMM_data.txt' # AMM: Atlantic Meridional Mode Index

    start_date = datetime(start_year, 1, 1, 0, 0, 0)
    end_date = datetime(end_year, 12, 1, 0, 0, 0)

    # Read in climate index data file specified by climate_index
    if climate_index=="oni" or climate_index=="ONI":
        df_climate_index = prepare_ONI(file_path_ONI, start_date, end_date)
        climate_index_name = 'oni'
    elif climate_index=="dmi" or climate_index=="DMI":
        df_climate_index = prepare_DMI(file_path_DMI, start_date, end_date)
        climate_index_name = 'dmi'
    elif climate_index=="amm" or climate_index=="AMM":
        df_climate_index = prepare_AMM(file_path_AMM, start_date, end_date)
        climate_index_name = 'amm'
    else:
        raise ValueError("Specified climate_index is not a valid climate index name.")
    
    # Read in and initialize the air temperature anomaly data
    dat = nc.Dataset(file_path_AIR)

    VAR1=dat.variables['air']
    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    time = dat.variables['time'][:]

    # Define the reference date: 1800-01-01 00:00:00. This is reference specified by NOAA's air.2m.mon.mean.nc file. 
    reference_date = datetime(1800, 1, 1, 0, 0, 0)

    dates = np.array([reference_date + timedelta(hours=int(h)) for h in time])
    start_time_ind = int(np.where(dates == start_date)[0][0])
    end_time_ind = int(int(np.where(dates == end_date)[0][0]) + 1)
    VAR1 = VAR1[start_time_ind:end_time_ind, :, :]

    VAR1_standard = np.empty_like(VAR1) # Initialize a new array to store the standardized data
    n_time, n_lat, n_long = VAR1.shape
    print("Original shape of air temp. data: ", n_time, n_lat, n_long)

    # Loop through each (lat, long) point and standardize the time series at each grid point
    print("Standardizing air temp. data...")
    for i in range(n_lat):
        if (i%100==0): 
            print("...", i)
        for j in range(n_long):
            time_series = VAR1[:, i, j]
            mean = np.mean(time_series)
            std = np.std(time_series)
            # Standardize the time series (avoid division by zero)
            if std != 0:
                VAR1_standard[:, i, j] = (time_series - mean) / std
            else:
                raise ValueError("std=0 at grid point: (", i, ",", j,")!")
            
    # Check if df_climate_index's and VAR1_standard's ts indicies are idential.
    ind_time = df_climate_index.index.strftime('%Y-%m-%d').to_numpy()
    vectorized_format = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))
    VAR1_time  = vectorized_format(dates[start_time_ind:end_time_ind])
    if not np.array_equal(ind_time, VAR1_time):
        raise ValueError("---The two date arrays of ind_time and VAR1_time (air temp.) are NOT identical.---")
    else:
        print("---The two date arrays of ind_time and VAR1_time (air temp.) are identical.---")

    # Compute correlation(air temp, climate index) at each grid point
    rho_all_months = np.empty((2, # index1: corr value; index2 = p-value
                        VAR1_standard.shape[1],
                        VAR1_standard.shape[2]))

    for i in range(n_lat):
        if (i%100==0): 
            print("...", i)
        for j in range(n_long):
            df_help = pd.DataFrame({
                'month': df_climate_index.index.month,
                'ind_ts': df_climate_index['ANOM'],
                'air_ts': VAR1_standard[:, i, j]})
            lag_string = 'ind_ts_lag' + str(num_lag) + 'm'
            df_help[lag_string] = df_help['ind_ts'].shift((num_lag))
            df_help = df_help.dropna()
            pearsonr_result = pearsonr(df_help[lag_string], df_help['air_ts'])
            rho_all_months[0,i,j] = pearsonr_result[0]
            rho_all_months[1,i,j] = pearsonr_result[1]

    rho_array = xr.DataArray(data = rho_all_months,
                            coords={
                                "lat": lat,
                                "lon": lon
                            },
                            dims = ["value", "lat", "lon"],
                            attrs=dict(
                                description="Rho, correlation between air temp and" + str(climate_index_name) + ". Indices: index1: corr value; index2 = p-value.",
                                climate_index_used = climate_index_name,
                                cor_calc_start_date = str(start_date),
                                cor_calc_end_date = str(end_date),
                                L_lag = num_lag)
                            )

    rho_array.to_netcdf(save_path)


# compute_psi_Hsiang2011(climate_index="amm",
#                        start_year=1980, end_year=2020, 
#                        num_lag=2, 
#                        num_R=3, 
#                        save_path="/Users/tylerbagwell/Desktop/psi_Hsiang2011_amm.nc")


# compute_gridded_correlation(climate_index="amm", 
#                             start_year=1980, end_year=2020, 
#                             num_lag=3, 
#                             save_path="/Users/tylerbagwell/Desktop/rho_airVSamm_lag3.nc")



def compute_psi_Callahan2023(climate_index, start_year, end_year, save_path):
    """
    Computes teleconnection strength (psi) between specified climate_index and air temperature 
    anomaly at all global grid points based on the method of Callahan 2023 (w/o population weighting).
    Output is a NetCDF file with description.
    """

    # Check if arguments have appropriate type:
    if not isinstance(start_year, int):
        raise TypeError(f"Expected an integer for start_year, but got {type(start_year).__name__}.")
    if not isinstance(end_year, int):
        raise TypeError(f"Expected an integer for end_year, but got {type(end_year).__name__}.")

    # Read in data files
    file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc' # Air temperature anomaly
    file_path_ONI = 'data/NOAA_ONI_data.txt' # ONI: Oceanic Nino Index
    file_path_DMI = 'data/NOAA_DMI_data.txt' # DMI: Dipole Mode Index
    file_path_AMM = 'data/NOAA_AMM_data.txt' # AMM: Atlantic Meridional Mode Index

    start_date = datetime(start_year, 1, 1, 0, 0, 0)
    end_date = datetime(end_year, 12, 1, 0, 0, 0)

    # Read in climate index data file specified by climate_index
    if climate_index=="oni" or climate_index=="ONI":
        df_climate_index = prepare_ONI(file_path_ONI, start_date, end_date)
        climate_index_name = 'oni'
    elif climate_index=="dmi" or climate_index=="DMI":
        df_climate_index = prepare_DMI(file_path_DMI, start_date, end_date)
        climate_index_name = 'dmi'
    elif climate_index=="amm" or climate_index=="AMM":
        df_climate_index = prepare_AMM(file_path_AMM, start_date, end_date)
        climate_index_name = 'amm'
    else:
        raise ValueError("Specified climate_index is not a valid climate index name.")
    
    # Read in and initialize the air temperature anomaly data
    dat = nc.Dataset(file_path_AIR)

    VAR1=dat.variables['air']
    lat = dat.variables['lat'][:]
    lon = dat.variables['lon'][:]
    time = dat.variables['time'][:]

    # Define the reference date: 1800-01-01 00:00:00.
    # This is reference specified by NOAA's air.2m.mon.mean.nc file metadata: "hours since 1800-01-01 00:00:00" 
    reference_date = datetime(1800, 1, 1, 0, 0, 0)

    dates = np.array([reference_date + timedelta(hours=int(h)) for h in time])
    start_time_ind = int(np.where(dates == start_date)[0][0])
    end_time_ind = int(int(np.where(dates == end_date)[0][0]) + 1)
    VAR1 = VAR1[start_time_ind:end_time_ind, :, :]

    VAR1_standard = np.empty_like(VAR1) # Initialize a new array to store the standardized data
    n_time, n_lat, n_long = VAR1.shape
    print("Original shape of air temp. data: ", n_time, n_lat, n_long)

    # Loop through each (lat, long) point and standardize the time series at each grid point
    #
    def standardize_monthly(data):
        data = np.array(data)
        n = len(data)
        months = np.arange(n) % 12  # Assign month indices 0-11
        means = np.array([data[months == m].mean() for m in range(12)])
        stds = np.array([data[months == m].std() for m in range(12)])
        standardized = (data - means[months]) / stds[months]
        return standardized.tolist()
    
    #
    def detrend_monthly(data):
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
    
    #
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

    print("Standardizing air temp. data...")
    for i in range(n_lat):
        if (i%100==0): 
            print("...", i)
        for j in range(n_long):
            VAR1_standard[:, i, j] = standardize_and_detrend_monthly(VAR1[:, i, j])
            

    # Check if df_climate_index's and VAR1_standard's ts indicies are idential.
    ind_time = df_climate_index.index.strftime('%Y-%m-%d').to_numpy()
    vectorized_format = np.vectorize(lambda x: x.strftime('%Y-%m-%d'))
    VAR1_time  = vectorized_format(dates[start_time_ind:end_time_ind])
    if not np.array_equal(ind_time, VAR1_time):
        raise ValueError("---The two date arrays of ind_time and VAR1_time (air temp.) are NOT identical.---")
    else:
        print("---The two date arrays of ind_time and VAR1_time (air temp.) are identical.---")

    # Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
    df_climate_index.index = pd.to_datetime(df_climate_index.index)     # Ensure 'date' to datetime and extract year & month
    df_climate_index['year'] = df_climate_index.index.year
    df_climate_index['month'] = df_climate_index.index.month

    dec_df = df_climate_index[df_climate_index['month'] == 12].copy() # prepare December data from previous year
    dec_df['year'] = dec_df['year'] + 1  # Shift to next year
    dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    jan_feb_df = df_climate_index[df_climate_index['month'].isin([1, 2])].copy() # prepare January and February data for current year
    jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
    feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

    yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

    yearly['avg_ANOM'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_DJF = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)
    
    # Compute monthly correlation and teleconnection (psi) at each grid point
    # computes correlations for each month from JUN(t-1) to AUG(t) with DJF index(t)
    corrs_array = np.empty((15,n_lat,n_long))
    psi = np.empty((n_lat,n_long))

    print("\nComputing psi array...")
    for i in range(n_lat):
        if (i%10==0): 
            print("...", i)
        for j in range(n_long):
            current_VAR = pd.DataFrame(data=VAR1_standard[:,i,j],
                                        index=dates[start_time_ind:end_time_ind],
                                        columns=['air'])
            current_VAR.index = pd.to_datetime(current_VAR.index)
            current_VAR['year'] = current_VAR.index.year
            current_VAR['month'] = current_VAR.index.month
            # iterate through the months
            for k in range(1,16,1):
                # jun-dec of year t-1
                if (k<=7):
                    var_ts = current_VAR[current_VAR['month'] == int(k+6-1)].copy()
                    var_ts['year'] = var_ts['year'] + 1  # Shift to next year
                # jan-aug of year t
                else:
                    var_ts = current_VAR[current_VAR['month'] == int(k-7)].copy()

                # compute correlations of yearly month, k, air anomaly with index 
                var_ts = pd.merge(var_ts, index_DJF, how='inner', on='year')
                corrs_array[int(k-1),i,j] = pearsonr(var_ts['air'], var_ts['avg_ANOM'])[0]

            corrs = pd.Series(corrs_array[:,i,j])
            rolling_avg = corrs.rolling(window=3, center=False).mean() ### BE AWARE OF CENTERING OF WINDOW!!!
            rolling_avg = np.abs(rolling_avg)
            max_corr = np.nanmax(rolling_avg)
            psi[i,j] = max_corr

    psi_array = xr.DataArray(data = psi,
                                coords={
                                "lat": lat,
                                "lon": lon
                            },
                            dims = ["lat", "lon"],
                            attrs=dict(
                                description="Psi, teleconnection strength via Callahan 2023 method.",
                                cor_calc_start_date = str(start_date),
                                cor_calc_end_date = str(end_date),
                                climate_index_used = climate_index_name)
                            )
    
    # Save psi_array as NetCDF file
    psi_array.to_netcdf(save_path)


# compute_psi_Callahan2023(climate_index='oni',
#                          start_year=1960, 
#                          end_year=2023,
#                          save_path="/Users/tylerbagwell/Desktop/psi_callahan_test_NOCENTER_detrended.nc")




# sys.exit()

###################################


start_year  = 1960
end_year    = 2020


file_path_AIR = '/Users/tylerbagwell/Desktop/air.2m.mon.mean.nc' # Air temperature anomaly
file_path_PREC = '/Users/tylerbagwell/Desktop/precip.mon.total.1x1.v2020.nc' # Air temperature anomaly
    

import xarray as xr

ds1 = xr.open_dataset(file_path_AIR)
ds2 = xr.open_dataset(file_path_PREC)

var1 = ds1['air']  # DataArray from the first dataset
var2 = ds2['precip']  # DataArray from the second dataset

# Access longitude and latitude coordinates
lon1 = ds1['lon']
lat1 = ds1['lat']
lon2 = ds2['lon']
lat2 = ds2['lat']

# Function to convert longitude from 0-360 to -180 to 180
def convert_longitude(ds):
    lon = ds['lon']
    lon = ((lon + 180) % 360) - 180
    ds = ds.assign_coords(lon=lon)
    return ds

# Apply conversion if necessary
if lon1.max() > 180:
    ds1 = convert_longitude(ds1)
if lon2.max() > 180:
    ds2 = convert_longitude(ds2)

ds1 = ds1.sortby('lon')
ds2 = ds2.sortby('lon')

ds1 = ds1.assign_coords(
    lon=np.round(ds1['lon'], decimals=2),
    lat=np.round(ds1['lat'], decimals=2)
)
ds2 = ds2.assign_coords(
    lon=np.round(ds2['lon'], decimals=2),
    lat=np.round(ds2['lat'], decimals=2)
)

# load index data
clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
                         end_date=datetime(end_year, 12, 1, 0, 0, 0))

common_lon  = np.intersect1d(ds1['lon'], ds2['lon']) #probably should check that this is not null
common_lat  = np.intersect1d(ds1['lat'], ds2['lat'])
common_time = np.intersect1d(ds1['time'], ds2['time'])
common_time = np.intersect1d(common_time, clim_ind.index.to_numpy())

ds1_common      = ds1.sel(time=common_time, lon=common_lon, lat=common_lat)
ds2_common      = ds2.sel(time=common_time, lon=common_lon, lat=common_lat)
clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]

var1_common = ds1_common['air']
var2_common = ds2_common['precip']

# Check shapes
print("var1_common shape:", var1_common.shape)
print("var2_common shape:", var2_common.shape)
print("clim_ind shape:   ", clim_ind_common.shape)

n_time, n_lat, n_long = var1_common.shape



# Verify that coordinates are identical
assert np.array_equal(var1_common['lon'], var2_common['lon'])
assert np.array_equal(var1_common['lat'], var2_common['lat'])
assert np.array_equal(var1_common['time'], var2_common['time'])
assert np.array_equal(var1_common['time'], clim_ind_common.index)
assert np.array_equal(var2_common['time'], clim_ind_common.index)


def standardize_and_detrend_monthly(data):
    data = np.array(data)
    n = len(data)
    months = np.arange(n) % 12  # Assign month indices 0-11
    means = np.array([data[months == m].mean() for m in range(12)])
    stds = np.array([data[months == m].std() for m in range(12)])

    standardized = (data - means[months]) #/ stds[months]

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
var2_std = np.empty_like(var2_common) # Initialize a new array to store the standardized data

var1_std = var1_common # DELETE LATER!!!!!!!!!
var2_std = var2_common # DELETE LATER!!!!!!!!!

# print("Standardizing air temp. data...")
# for i in range(n_lat):
#     if (i%25==0): 
#         print("...", i)
#     for j in range(n_long):
#         var1_std[:, i, j] = standardize_and_detrend_monthly(var1_common[:, i, j])
#         has_nan = np.isnan(var2_common[:, i, j]).any()
#         if (has_nan==False):
#             var2_std[:, i, j] = standardize_and_detrend_monthly(var2_common[:, i, j])
#         else: 
#             var2_std[:, i, j] = var2_common[:, i, j]

# Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
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
index_DJF = yearly[['year', 'avg_ANOM']].sort_values('year').reset_index(drop=True)

# Compute monthly correlation and teleconnection (psi) at each grid point
# computes correlations for each month from JUN(t-1) to AUG(t) with DJF index(t)
corrs_array_1 = np.empty((15,n_lat,n_long))
corrs_array_2 = np.empty((15,n_lat,n_long))
psi = np.empty((n_lat,n_long))

print("\nComputing psi array...")
for i in range(n_lat):
    if (i%10==0): 
        print("...", i)
    for j in range(n_long):
        current_vars = pd.DataFrame(data=var1_std[:,i,j],
                                    index=var1_common['time'], #need to use var1_common since it still contains the time data
                                    columns=['air'])
        current_vars['precip'] = np.array(var2_std[:,i,j])
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
            var_ts = pd.merge(var_ts, index_DJF, how='inner', on='year')
    
            has_nan = var_ts['precip'].isna().any()
            if has_nan==False:
                partial_corr_1 = partial_corr(data=var_ts, x='air', y='avg_ANOM', covar='precip')['r'].values[0]
                partial_corr_2 = partial_corr(data=var_ts, x='precip', y='avg_ANOM', covar='air')['r'].values[0]
                corrs_array_1[int(k-1),i,j] = partial_corr_1
                corrs_array_2[int(k-1),i,j] = partial_corr_2
            else:
                corrs_array_1[int(k-1),i,j] = np.nan
                corrs_array_2[int(k-1),i,j] = np.nan

        corrs1 = pd.Series(corrs_array_1[:,i,j])
        corrs2 = pd.Series(corrs_array_2[:,i,j])

        has_nan = corrs1.isna().any()
        if has_nan==False:
            # var1
            rolling_avg1 = corrs1.rolling(window=3, center=False).mean() ### BE AWARE OF CENTERING OF WINDOW!!!
            rolling_avg1 = np.abs(rolling_avg1)
            max_corr1 = np.nanmax(rolling_avg1)
            # var2
            rolling_avg2 = corrs2.rolling(window=3, center=False).mean() ### BE AWARE OF CENTERING OF WINDOW!!!
            rolling_avg2 = np.abs(rolling_avg2)
            max_corr2 = np.nanmax(rolling_avg2)
            # compute teleconnection (psi)
            psi[i,j] = max_corr1 + max_corr2
        else:
            psi[i,j] = np.nan

psi_array = xr.DataArray(data = psi,
                            coords={
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["lat", "lon"],
                        attrs=dict(
                            description="Psi, teleconnection strength via Callahan 2023 method.",
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = 'NINO3')
                        )

psi_array.to_netcdf('/Users/tylerbagwell/Desktop/psi_callahan_NINO3.nc')