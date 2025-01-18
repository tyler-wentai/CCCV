import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
import geopandas as gpd
from prepare_index import *

print('\n\nSTART ---------------------\n')

#
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
    
    return ann_ind

#
def compute_bymonth_corr_map(ds_in, climate_index, annualized_index):
    var_str = list(ds_in.data_vars)[0]
    n_time, n_lat, n_long = ds_in[var_str].shape

    n_months = 12
    corr_monthly = np.empty((n_months, n_lat, n_long))

    def pearsonr_func(a, b):
        return pearsonr(a, b)

    for i in range(1,n_months+1):
        print("...[var:", var_str, "] Computing tropical month:", i, "of", n_months)
        if (climate_index == 'nino3' or climate_index == 'nino34'): 
            ### NINO3 or NINO3.4 (tropical year from June y_{t-1} to May y_{t})
            if (i<=7): 
                m = i + 5
                y = 1
            else:
                m = i - 7
                y = 0
        elif (climate_index == 'dmi'):
            ### DMI (tropical year from March y_{t} to February y_{t+1})
            if (i<=10): 
                m = i + 2
                y = 0
            else:
                m = i - 10
                y = -1
        elif (climate_index == 'ani'): 
            ### ANI (tropical year from March y_{t} to February y_{t+1})
            if (i<=10): 
                m = i + 2
                y = 0
            else:
                m = i - 10
                y = -1
        else:
            raise ValueError("Specified 'climate_index' not found...")

        # Convert "year" + month to a DatetimeIndex (assuming day=1)
        ann_help = annualized_index.copy()
        ann_help['date'] = pd.to_datetime({
            'year': annualized_index['year'] + y,        # Note: + y is here to help with computing corr's of months in the previous tropical year or present year
            'month': m,
            'day': 1
        })

        ann_help.set_index('date', inplace=True)
        ann_help.drop(columns='year', inplace=True)

        ann_ind_ts = xr.DataArray(
            ann_help["ann_ind"],
            coords=[ann_help.index],   # use the pandas DatetimeIndex as the coords
            dims=["valid_time"])        # name the dimension 'time'

        var_aligned, ind_aligned = xr.align(ds_in[var_str], ann_ind_ts, join="inner")

        # standardize the align variable data
        mean_data           = var_aligned.mean(dim='valid_time', skipna=True)
        std_data            = var_aligned.std(dim='valid_time', skipna=True)
        var_standardized    = (var_aligned - mean_data) / std_data
        var_standardized    = var_standardized.where(std_data != 0, 0) # handle the special case: if std == 0 at a grid cell, set all times there to 0

        # compute correlations and their p-values
        corr_map, pval_map = xr.apply_ufunc(
            pearsonr_func,
            var_standardized,       # first input
            ind_aligned,            # second input
            input_core_dims=[["valid_time"], ["valid_time"]],   # dimension(s) over which to compute
            output_core_dims=[[], []],                          # correlation, p-value are scalars per grid
            vectorize=True,                                     # run function for each (lat, lon) point
        )

        # set all correlations to zero if its p-value is less than the threshold
        threshold = 0.05
        pval_mask = pval_map < threshold
        pval_mask = pval_mask.values

        corr_results = corr_map.values
        corr_results = np.where(pval_mask, corr_results, 0)

        # save to corr_final
        corr_monthly[(i-1),:,:] = corr_results

    return corr_monthly

#
def compute_teleconnection(var1_path, var2_path, resolution, climate_index, start_year, end_year, plot_psi=False):
    ""
    ds1 = xr.open_dataset(var1_path)
    var1_str = list(ds1.data_vars)[0]

    ds2 = xr.open_dataset(var2_path)
    var2_str = list(ds2.data_vars)[0]
    
    print("1st variable accessed is: ", var1_str)
    print("2nd variable accessed is: ", var2_str)

    # change dates to time format:
    dates1 = pd.to_datetime(ds1['date'].astype(str), format='%Y%m%d')
    ds1 = ds1.assign_coords(date=dates1)
    ds1 = ds1.rename({'date': 'valid_time'})

    dates2 = pd.to_datetime(ds2['date'].astype(str), format='%Y%m%d')
    ds2 = ds2.assign_coords(date=dates2)
    ds2 = ds2.rename({'date': 'valid_time'})

    # Access longitude and latitude coordinates
    lon1 = ds1['longitude']
    lat1 = ds1['latitude']

    lon2 = ds2['longitude']
    lat2 = ds2['latitude']

    lat_int_mask1 = (lat1 % resolution == 0)
    lon_int_mask1 = (lon1 % resolution == 0)
    ds1 = ds1.sel(latitude=lat1[lat_int_mask1], longitude=lon1[lon_int_mask1])

    lat_int_mask2 = (lat2 % resolution == 0)
    lon_int_mask2 = (lon2 % resolution == 0)
    ds2 = ds2.sel(latitude=lat2[lat_int_mask2], longitude=lon2[lon_int_mask2])

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

    if lon2.max() > 180:
        ds2 = convert_longitude(ds2)
    ds2 = ds2.sortby('longitude')

    # ENSURE THAT ds1 AND ds2 ARE ALIGNED IN (valid_time, latitude, longitude)
    ds1_aligned, ds2_aligned = xr.align(ds1, ds2, join="inner")

    print("Aligned 1st variable shape:", ds1_aligned[var1_str].shape)
    print("Aligned 2nd variable shape:", ds2_aligned[var2_str].shape)

    ### COMPUTE ANNUALIZED CLIMATE INDEX
    annualized_index = compute_annualized_index(climate_index, start_year, end_year)

    ### COMPUTE MONTHLY CORRELATION MAPS FOR VAR1 and VAR2
    corr_array1 = compute_bymonth_corr_map(ds1_aligned, climate_index, annualized_index)
    corr_array2 = compute_bymonth_corr_map(ds2_aligned, climate_index, annualized_index)

    ### COMPUTE TELECONNECTION STRENGTH
    telecon_var1 = np.abs(corr_array1)
    telecon_var1 = np.sum(telecon_var1, axis=0)

    telecon_var2 = np.abs(corr_array2)
    telecon_var2 = np.sum(telecon_var2, axis=0)

    telecon_total = telecon_var1 + telecon_var2

    psi = xr.DataArray(telecon_total,
                       coords = {"latitude":  ds1['latitude'],
                                 "longitude": ds1['longitude']
                                 },
                        dims = ["latitude", "longitude"])
    
    ### PLOT TELECONNECTION
    if (plot_psi == True):
        path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
        gdf = gpd.read_file(path_land)

        fig, ax = plt.subplots(figsize=(10, 6))
        psi.plot(
            x="longitude", 
            y="latitude",
            ax=ax,
            cmap="Reds",  # a diverging colormap is nice for correlations
        )
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        ax.set_title("Teleconnection with Country Outlines")
        plt.show()






compute_teleconnection(var1_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_t2m_raw.nc', 
                       var2_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc',
                       resolution = 2.0,
                       climate_index = 'dmi', 
                       start_year = 1980, 
                       end_year = 2023,
                       plot_psi = True)