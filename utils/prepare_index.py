
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import sys

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
def prepare_NINO34(file_path, start_date, end_date):
    """
    Prepare NINO3.4 index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino34/)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nino34 = pd.read_csv(file_path, sep=r'\s+', skiprows=1, skipfooter=7, header=None, engine='python')
    year_start = int(nino34.iloc[0,0])
    nino34 = nino34.iloc[:,1:nino34.shape[1]].values.flatten()
    df_nino34 = pd.DataFrame(nino34)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_nino34.shape[0], freq='MS')
    df_nino34.index = date_range
    df_nino34.rename_axis('date', inplace=True)
    df_nino34.columns = ['ANOM']

    start_ts_l = np.where(df_nino34.index == start_date)[0]
    end_ts_l = np.where(df_nino34.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of NINO3.4 index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of NINO3.4 index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_nino34 = df_nino34.iloc[start_ts_ind:end_ts_ind]

    return df_nino34


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


#
def prepare_ANI(file_path, start_date, end_date):
    """
    Prepare Atlantic Nino Index (ANI) data as pd.Data.Frame from csv file with
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    ani = pd.read_csv(file_path)
    ani['time'] = pd.to_datetime(
        ani['time'],
        format='%Y-%m-%d %H:%M:%S.%f', 
        )
    ani['time'] = ani['time'].apply(lambda dt: dt.replace(day=1))
    ani['time'] = ani['time'].dt.floor('D')
    ani = ani.drop('month', axis=1)
    year_start = int(ani['time'].dt.year.min())
    ani = ani.iloc[:,1:ani.shape[1]].values.flatten()
    df_ani = pd.DataFrame(ani)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=ani.shape[0], freq='MS')
    df_ani.index = date_range
    df_ani.rename_axis('date', inplace=True)
    df_ani.columns = ['ANOM']

    start_ts_l = np.where(df_ani.index == start_date)[0]
    end_ts_l = np.where(df_ani.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of ANI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of ANI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_ani = df_ani.iloc[start_ts_ind:end_ts_ind]

    return df_ani

#
def prepare_IOD_CAI(file_path, start_date, end_date):
    """
    Prepare Indian Ocean Dipole data using Cai et al. method as pd.Data.Frame from csv file with
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    iod = pd.read_csv(file_path)
    iod['time'] = pd.to_datetime(
        iod['time'],
        format='%Y', 
        )
    # iod['time'] = iod['time'].apply(lambda dt: dt.replace(day=1))
    iod['time'] = iod['time'].dt.floor('D')
    # iod = iod.drop('month', axis=1)
    year_start = int(iod['time'].dt.year.min())
    # iod = iod.iloc[:,1:iod.shape[1]].values.flatten()
    df_iod = pd.DataFrame(iod)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=iod.shape[0], freq='YS')

    df_iod.index = date_range
    df_iod.rename_axis('date', inplace=True)
    df_iod.drop('time', axis=1, inplace=True)
    df_iod.columns = ['ANOM_LD', 'ANOM_QD']  # LD: Linearly detrended, QD: Quadratically detrended

    start_ts_l = np.where(df_iod.index == start_date)[0]
    end_ts_l = np.where(df_iod.index == end_date)[0]

    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of IOD index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of IOD index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_iod = df_iod.iloc[start_ts_ind:end_ts_ind]

    print('NOTE: prepare_IOD_CAI is returning the QUADRATICALLY DETRENDED IOD index.')
    df_iod.drop('ANOM_LD', axis=1, inplace=True)
    df_iod.rename(columns={'ANOM_QD': 'ANOM'}, inplace=True)

    # print('NOTE: prepare_IOD_CAI is returning the LINEARLY DETRENDED IOD index.')
    # df_iod.drop('ANOM_QD', axis=1, inplace=True)
    # df_iod.rename(columns={'ANOM_LD': 'ANOM'}, inplace=True)

    return df_iod

#
def prepare_Eindex(file_path, start_date, end_date):
    """
    Prepare NINO E-Index (NEI) data as pd.Data.Frame from csv file with
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nei = pd.read_csv(file_path)
    nei['time'] = pd.to_datetime(
        nei['time'],
        format='%Y-%m-%d %H:%M:%S.%f', 
        )
    nei['time'] = nei['time'].apply(lambda dt: dt.replace(day=1))
    nei['time'] = nei['time'].dt.floor('D')
    nei = nei.drop('month', axis=1)
    year_start = int(nei['time'].dt.year.min())
    nei = nei['Eindex'].values.flatten()

    df_nei = pd.DataFrame(nei)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=nei.shape[0], freq='MS')
    df_nei.index = date_range
    df_nei.rename_axis('date', inplace=True)
    df_nei.columns = ['ANOM']


    start_ts_l = np.where(df_nei.index == start_date)[0]
    end_ts_l = np.where(df_nei.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of ANI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of ANI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_nei = df_nei.iloc[start_ts_ind:end_ts_ind]

    return df_nei


#
def prepare_Cindex(file_path, start_date, end_date):
    """
    Prepare NINO C-Index (NEI) data as pd.Data.Frame from csv file with
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nci = pd.read_csv(file_path)
    nci['time'] = pd.to_datetime(
        nci['time'],
        format='%Y-%m-%d %H:%M:%S.%f', 
        )
    nci['time'] = nci['time'].apply(lambda dt: dt.replace(day=1))
    nci['time'] = nci['time'].dt.floor('D')
    nci = nci.drop('month', axis=1)
    year_start = int(nci['time'].dt.year.min())
    nci = nci['Cindex'].values.flatten()

    df_nci = pd.DataFrame(nci)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=nci.shape[0], freq='MS')
    df_nci.index = date_range
    df_nci.rename_axis('date', inplace=True)
    df_nci.columns = ['ANOM']


    start_ts_l = np.where(df_nci.index == start_date)[0]
    end_ts_l = np.where(df_nci.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of ANI index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of ANI index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_nci = df_nci.iloc[start_ts_ind:end_ts_ind]

    return df_nci


#
def compute_annualized_NINO3_index(start_year, end_year, save_path=False):
    """
    Computes the annualized NINO3 index via the average of the index of DEC(t-1),JAN(t),FEB(t) based on
    the method of Callahan 2023
    """
    # load index data
    clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    dec_df = clim_ind[clim_ind['month'] == 12].copy() # prepare December data from previous year
    dec_df['year'] = dec_df['year'] + 1  # Shift to next year
    dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    jan_feb_df = clim_ind[clim_ind['month'].isin([1, 2])].copy() # prepare January and February data for current year
    jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
    feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

    yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

    yearly['INDEX'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    # nov_dec_df = clim_ind[clim_ind['month'].isin([11,12])].copy() # prepare November, December, and January data 
    # nov     = nov_dec_df[nov_dec_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})
    # dec     = nov_dec_df[nov_dec_df['month'] == 12][['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    # jan_df = clim_ind[clim_ind['month'].isin([1, 2])].copy() # prepare January data
    # jan_df['year'] = jan_df['year'] - 1  # Shift to past year
    # jan     = jan_df[jan_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})

    # yearly = pd.merge(nov, dec, on='year', how='inner') # merge 
    # yearly = pd.merge(yearly, jan, on='year', how='inner') # merge 

    # yearly['INDEX'] = yearly[['NOV_ANOM', 'DEC_ANOM', 'JAN_ANOM']].mean(axis=1) # Calculate the average NDJ ANOM value
    # index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    if (save_path!=False):
        np.save(save_path, index_yrAVG)

    return index_yrAVG

#
def compute_annualized_DMI_index(start_year, end_year, save_path=False):
    """
    Computes the annualized DMI index via the average of the index of SEP(t),OCT(t),NOV(t) inspired by
    the method of Callahan 2023
    """
    # load index data
    clim_ind = prepare_DMI(file_path='data/NOAA_DMI_data.txt',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of SEP(t),OCT(t),NOV(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    sep_oct_nov_df = clim_ind[clim_ind['month'].isin([9, 10, 11])].copy() # prepare September, October, and November data for current year
    sep     = sep_oct_nov_df[sep_oct_nov_df['month'] == 9][['year', 'ANOM']].rename(columns={'ANOM': 'SEP_ANOM'})
    oct     = sep_oct_nov_df[sep_oct_nov_df['month'] == 10][['year', 'ANOM']].rename(columns={'ANOM': 'OCT_ANOM'})
    nov     = sep_oct_nov_df[sep_oct_nov_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})

    yearly = pd.merge(sep, oct, on='year', how='inner') # merge September, October data
    yearly = pd.merge(yearly, nov, on='year', how='inner') # merge December, January, and February data

    yearly['INDEX'] = yearly[['SEP_ANOM', 'OCT_ANOM', 'NOV_ANOM']].mean(axis=1) # Calculate the average SON ANOM value
    index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)


    if (save_path!=False):
        np.save(save_path, index_yrAVG)

    return index_yrAVG


#
def compute_annualized_ANI_index(start_year, end_year, save_path=False):
    """
    Computes the annualized ANI (Atlantic Nino Index) via the average of the index of JUN(t),JUL(t),AUG(t) inspired by
    the method of Callahan 2023.
    """
    # load index data
    clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of JUN(t),JUL(t),AUG(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    jun_jul_aug_df = clim_ind[clim_ind['month'].isin([6, 7, 8])].copy() # prepare June, July, and August data for current year
    jun     = jun_jul_aug_df[jun_jul_aug_df['month'] == 6][['year', 'ANOM']].rename(columns={'ANOM': 'JUN_ANOM'})
    jul     = jun_jul_aug_df[jun_jul_aug_df['month'] == 7][['year', 'ANOM']].rename(columns={'ANOM': 'JUL_ANOM'})
    aug     = jun_jul_aug_df[jun_jul_aug_df['month'] == 8][['year', 'ANOM']].rename(columns={'ANOM': 'AUG_ANOM'})

    yearly = pd.merge(jun, jul, on='year', how='inner') # merge June, July data
    yearly = pd.merge(yearly, aug, on='year', how='inner') # merge June, July, and August data

    yearly['INDEX'] = yearly[['JUN_ANOM', 'JUL_ANOM', 'AUG_ANOM']].mean(axis=1) # Calculate the average JJA ANOM value
    index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    if (save_path!=False):
        np.save(save_path, index_yrAVG)

    return index_yrAVG


#
def compute_annualized_EEI_index(start_year, end_year, save_path=False):
    """
    Computes the annualized ENSO E-Index via the average of the index of DEC(t-1),JAN(t),FEB(t) inspired by
    the method of Callahan 2023.
    """
    # load index data
    clim_ind = prepare_Eindex(file_path='data/CE_index.csv',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    dec_df = clim_ind[clim_ind['month'] == 12].copy() # prepare December data from previous year
    dec_df['year'] = dec_df['year'] + 1  # Shift to next year
    dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    jan_feb_df = clim_ind[clim_ind['month'].isin([1, 2])].copy() # prepare January and February data for current year
    jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
    feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

    yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

    yearly['INDEX'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    return index_yrAVG


#
def compute_annualized_ECI_index(start_year, end_year, save_path=False):
    """
    Computes the annualized ENSO C-Index via the average of the index of DEC(t-1),JAN(t),FEB(t) inspired by
    the method of Callahan 2023.
    """
    # load index data
    clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    dec_df = clim_ind[clim_ind['month'] == 12].copy() # prepare December data from previous year
    dec_df['year'] = dec_df['year'] + 1  # Shift to next year
    dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    jan_feb_df = clim_ind[clim_ind['month'].isin([1, 2])].copy() # prepare January and February data for current year
    jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
    feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

    yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

    yearly['INDEX'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_yrAVG = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    return index_yrAVG


