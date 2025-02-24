import numpy as np
import pandas as pd
from datetime import datetime
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

def compute_annualized_index(climate_index, start_year, end_year):
    if (climate_index == 'nino34'):
        clim_ind = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
    elif (climate_index == 'nino3'):
        clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
    elif (climate_index == 'eci'):
        clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
    elif (climate_index == 'dmi'):
        clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
    elif (climate_index == 'ani'):
        clim_ind = prepare_ANI(file_path='data/Atlantic_NINO.csv',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
    else:
        raise ValueError("Specified 'climate_index' not found...")
    
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    if (climate_index == 'nino3' or climate_index == 'nino34'): ### NINO3 or NINO3.4
        # 1) Add a 'DJF_year' column that treats December as belonging to the *next* year
        clim_ind['DJF_year'] = clim_ind.index.year
        clim_ind.loc[clim_ind.index.month == 12, 'DJF_year'] += 1

        # 2) Filter for only DJF months (12, 1, 2)
        djf = clim_ind[clim_ind.index.month.isin([12, 1, 2])]
        #djf = clim_ind[clim_ind.index.month.isin([5, 6, 7, 8, 9, 10, 11, 12])]

        # 3) Group by 'DJF_year' and compute the mean anomaly to obtain annualized index values
        ann_ind = djf.groupby('DJF_year').ANOM.agg(['mean', 'count']).reset_index()
        ann_ind = ann_ind[ann_ind['count'] == 3]    # Only keep years with all three months of data
        ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'DJF_year': 'year'})
        ann_ind = ann_ind.drop(['count'], axis=1)
    elif (climate_index == 'dmi'): ### DMI
        # 1) Add a 'SON_year' column
        clim_ind['SON_year'] = clim_ind.index.year

        # 2) Filter for only SON months (9, 10, 11)
        son = clim_ind[clim_ind.index.month.isin([9, 10, 11])]

        # 3) Group by 'SON_year' and compute the mean anomaly to obtain annualized index values
        ann_ind = son.groupby('SON_year').ANOM.agg(['mean', 'count']).reset_index()
        ann_ind = ann_ind[ann_ind['count'] == 3]    # Only keep years with all three months of data
        ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'SON_year': 'year'})
        ann_ind = ann_ind.drop(['count'], axis=1)
    elif (climate_index == 'ani'): ### ANI
        # 1) Add a 'JJA_year' column
        clim_ind['JJA_year'] = clim_ind.index.year

        # 2) Filter for only JJA months (6, 7, 8)
        jja = clim_ind[clim_ind.index.month.isin([6, 7, 8])]

        # 3) Group by 'JJA_year' and compute the mean anomaly to obtain annualized index values
        ann_ind = jja.groupby('JJA_year').ANOM.agg(['mean', 'count']).reset_index()
        ann_ind = ann_ind[ann_ind['count'] == 3]    # Only keep years with all three months of data
        ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'JJA_year': 'year'})
        ann_ind = ann_ind.drop(['count'], axis=1)
    else:
        raise ValueError("Specified 'climate_index' not found...")
    
    ann_ind = ann_ind.rename(columns={'ann_ind': 'cindex'}) # cindex: climate index
    return ann_ind