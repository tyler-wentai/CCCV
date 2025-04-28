import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
import geopandas as gpd
from pingouin import partial_corr
from oldcode.prepare_index import *

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
    elif (climate_index == 'iod_cai'):
        clim_ind = prepare_IOD_CAI(file_path='data/IOD_index_Cai.csv',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 1, 1, 0, 0, 0)) # Note: iod_cai end_date must start at January 1st unlike other indices.
    else:
        raise ValueError("Specified 'climate_index' not found...")
    
    clim_ind['year'] = clim_ind.index.year

    if (climate_index == 'nino3' or climate_index == 'nino34'): ### NINO3 or NINO3.4
        clim_ind['month'] = clim_ind.index.month
        # 1) Add a 'DJF_year' column that treats December as belonging to the *next* year
        clim_ind['DJF_year'] = clim_ind.index.year
        # clim_ind.loc[clim_ind.index.month == 12, 'DJF_year'] += 1

        # 2) Filter for only DJF months (12, 1, 2)
        # djf = clim_ind[clim_ind.index.month.isin([12, 1, 2])]
        djf = clim_ind[clim_ind.index.month.isin([5, 6, 7, 8, 9, 10, 11, 12])]

        # 3) Group by 'DJF_year' and compute the mean anomaly to obtain annualized index values
        ann_ind = djf.groupby('DJF_year').ANOM.agg(['mean', 'count']).reset_index()
        ann_ind = ann_ind[ann_ind['count'] == 8]    # Only keep years with all three months of data
        ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'DJF_year': 'year'})
        ann_ind = ann_ind.drop(['count'], axis=1)
    elif (climate_index == 'dmi'): ### DMI
        clim_ind['month'] = clim_ind.index.month
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
        clim_ind['month'] = clim_ind.index.month
        # 1) Add a 'JJA_year' column
        clim_ind['JJA_year'] = clim_ind.index.year

        # 2) Filter for only JJA months (6, 7, 8)
        jja = clim_ind[clim_ind.index.month.isin([6, 7, 8])]

        # 3) Group by 'JJA_year' and compute the mean anomaly to obtain annualized index values
        ann_ind = jja.groupby('JJA_year').ANOM.agg(['mean', 'count']).reset_index()
        ann_ind = ann_ind[ann_ind['count'] == 3]    # Only keep years with all three months of data
        ann_ind = ann_ind.rename(columns={'mean': 'ann_ind', 'JJA_year': 'year'})
        ann_ind = ann_ind.drop(['count'], axis=1)
    elif (climate_index == 'iod_cai'): ### IOD via Cai et al. (2011) method computed by Xinyue
        clim_ind = clim_ind.rename(columns={'ANOM': 'ann_ind'})
        clim_ind.reset_index(drop=True, inplace=True)
        ann_ind = clim_ind.copy()
    else:
        raise ValueError("Specified 'climate_index' not found...")
    
    return ann_ind

#
def compute_bymonth_partialcorr_map(ds1_in, ds2_in, climate_index, annualized_index, enso_index):
    var1_str = ds1_in.name
    var2_str = ds2_in.name

    var1, var2 = xr.align(ds1_in, ds2_in, join="inner")

    n_time, n_lat, n_long = var1.shape

    # n_months = 12
    n_months = 8
    corr_monthly = np.empty((n_months, n_lat, n_long))

    def partial_corr_func(x, y, z1, z_enso, climate_index):
        """
        Computes partial correlation between arrays a and b, controlling for array c.
        Returns (r, pval).
        """
        # Convert arrays to a DataFrame for pingouin
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z1': z1,
            'z_enso': z_enso
        })

        # check for nans
        if df['y'].isna().any() or df['z1'].isna().any():
            r_val = np.nan
            p_val = np.nan
            return r_val, p_val

        if (climate_index == 'nino3' or climate_index == 'nino34'): 
            result = partial_corr(data=df, x='x', y='y', covar='z1', method='pearson')
        else:
            # Compute partial correlation with pingouin
            result = partial_corr(data=df, x='x', y='y', covar=['z1', 'z_enso'], method='pearson')

        # Extract r and p-values (first row, since partial_corr returns a DataFrame)
        r_val = result['r'].iloc[0]
        p_val = result['p-val'].iloc[0]
        
        return r_val, p_val

    for i in range(1,n_months+1):
        print("...(var:", var1_str, ", covar:", var2_str, "] Computing tropical month:", i, "of", n_months)
        if (climate_index == 'nino3' or climate_index == 'nino34'): 
            ### NINO3 or NINO3.4 (tropical year from June y_{t-1} to May y_{t})
            if (i<=7): 
                m = i + 5
                y = 1
            else:
                m = i - 7
                y = 0
        elif (climate_index == 'dmi' or climate_index == 'iod_cai'):
            ### DMI (tropical year from March y_{t} to February y_{t+1})
            if (i<=10): 
                m = i + 2
                # m = i + 4
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
        
        # compute NINO3.4 index time series
        enso_help = enso_index.copy()
        enso_help['date'] = pd.to_datetime({
            'year': enso_index['year'] + y,        # Note: + y is here to help with computing corr's of months in the previous tropical year or present year
            'month': m,
            'day': 1
        })

        enso_help.set_index('date', inplace=True)
        enso_help.drop(columns='year', inplace=True)

        enso_ind_ts = xr.DataArray(
            enso_help["ann_ind"],
            coords=[enso_help.index],   # use the pandas DatetimeIndex as the coords
            dims=["valid_time"])        # name the dimension 'time'
        
        # ensure data are aligned
        ind_aligned, enso_aligned = xr.align(ann_ind_ts, enso_ind_ts, join="inner")
        var1_aligned, ind_aligned = xr.align(var1, ind_aligned, join="inner")
        var2_aligned, ind_aligned = xr.align(var2, ind_aligned, join="inner")
        ind_aligned, enso_aligned = xr.align(ind_aligned, enso_aligned, join="inner")

        # standardize the aligned variable data
        mean1_data           = var1_aligned.mean(dim='valid_time', skipna=True)
        std1_data            = var1_aligned.std(dim='valid_time', skipna=True)
        var1_standardized    = (var1_aligned - mean1_data) / std1_data
        var1_standardized    = var1_standardized.where(std1_data != 0, 0) # handle the special case: if std == 0 at a grid cell, set all times there to 0

        mean2_data           = var2_aligned.mean(dim='valid_time', skipna=True)
        std2_data            = var2_aligned.std(dim='valid_time', skipna=True)
        var2_standardized    = (var2_aligned - mean2_data) / std2_data
        var2_standardized    = var2_standardized.where(std2_data != 0, 0) # handle the special case: if std == 0 at a grid cell, set all times there to 0

        # detrend the aligned variable data
        degree = 1
        var1_standardized = var1_standardized.assign_coords(
            time_numeric=(var1_standardized.valid_time - np.datetime64('1940-01-01')) / np.timedelta64(1, 's')).swap_dims({'valid_time': 'time_numeric'})
        fit1 = var1_standardized.polyfit(dim='time_numeric', deg=degree)
        trend1 = xr.polyval(var1_standardized.time_numeric, fit1.polyfit_coefficients)
        detrended1 = var1_standardized - trend1
        detrended1 = detrended1.swap_dims({'time_numeric':'valid_time'})

        var2_standardized = var2_standardized.assign_coords(
            time_numeric=(var2_standardized.valid_time - np.datetime64('1940-01-01')) / np.timedelta64(1, 's')).swap_dims({'valid_time': 'time_numeric'})
        fit2 = var2_standardized.polyfit(dim='time_numeric', deg=degree)
        trend2 = xr.polyval(var2_standardized.time_numeric, fit2.polyfit_coefficients)
        detrended2 = var2_standardized - trend2
        detrended2 = detrended2.swap_dims({'time_numeric':'valid_time'})

        print("......", detrended1.shape)
        print("......", detrended2.shape)
        print("......", ind_aligned.shape)

        # compute correlations and their p-values
        corr_map, pval_map = xr.apply_ufunc(
            partial_corr_func,
            ind_aligned,                    # first  input (x),  the climate index
            detrended1,              # second input (y),  the variable
            detrended2,              # third  input (z1), the first covariate to control for
            enso_aligned,                   # fourth input (z_enso), the second covariate to control for
            input_core_dims=[["valid_time"], ["valid_time"], ["valid_time"], ["valid_time"]],
            output_core_dims=[[], []],  # both correlation and p-value are scalars per lat/lon
            vectorize=True,
            kwargs={"climate_index": climate_index} # the climate index string
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
def compute_teleconnection(nc_path, save_path, nskip, climate_index, start_year, end_year, plot_psi=False):
    ""

    ds = xr.open_dataset(nc_path)
    ds = ds.isel(latitude=slice(0, None, int(nskip)), longitude=slice(0, None, int(nskip)))

    var1_str = list(ds.data_vars)[0]
    var2_str = list(ds.data_vars)[1]

    ds1 = ds[var1_str]
    ds2 = ds[var2_str]

    print("1st variable accessed is: ", var1_str)
    print("2nd variable accessed is: ", var2_str)

    # change dates to time format:
    # NOTE: For some reason the ERA5 tp dataset gives dates at 6AM and not 12AM, so we need these two lines below to align
    ds1 = ds1.assign_coords(valid_time=ds1.valid_time.dt.floor('D'))
    ds2 = ds2.assign_coords(valid_time=ds2.valid_time.dt.floor('D'))

    # Access longitude and latitude coordinates
    lon1 = ds1['longitude']
    lat1 = ds1['latitude']

    lon2 = ds2['longitude']
    lat2 = ds2['latitude']

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

    print("Aligned 1st variable shape:", ds1.shape)
    print("Aligned 2nd variable shape:", ds2.shape)

    ### COMPUTE ANNUALIZED CLIMATE INDEX
    annualized_index = compute_annualized_index(climate_index, start_year, end_year)
    enso_index       = compute_annualized_index("nino34", start_year, end_year)

    ### COMPUTE MONTHLY CORRELATION MAPS FOR VAR1 and VAR2
    corr_array1 = compute_bymonth_partialcorr_map(ds1, ds2, climate_index, annualized_index, enso_index)
    corr_array2 = compute_bymonth_partialcorr_map(ds2, ds1, climate_index, annualized_index, enso_index)

    ### COMPUTE TELECONNECTION STRENGTH
    telecon_var1 = np.abs(corr_array1)
    telecon_var1 = np.sum(telecon_var1, axis=0)

    telecon_var2 = np.abs(corr_array2)
    telecon_var2 = np.sum(telecon_var2, axis=0)

    # telecon_var1 = np.sum(corr_array1, axis=0)
    # telecon_var1 = np.abs(telecon_var1)

    # telecon_var2 = np.sum(corr_array2, axis=0)
    # telecon_var2 = np.abs(telecon_var2)

    telecon_total = telecon_var1 + telecon_var2

    ### SAVE TELECONNECTION STRENGTH TO NETCDF
    psi_str = "Teleconnection strength (psi) for" + climate_index + "with" +\
        var1_str + "and" + var2_str + "computed by partial correlation."
    psi = xr.DataArray(telecon_total,
                       coords = {"latitude":  ds1['latitude'],
                                 "longitude":  ds1['longitude']
                                 },
                        dims = ["latitude", "longitude"],
                        attrs=dict(
                            description=psi_str,
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = climate_index)
                            )
    
    save_path_help = save_path + "/psi_" + climate_index + "_LAND_nskip{:.1f}".format(nskip) + "_" +\
        str(start_year) + str(end_year) + "_12months.nc"
    psi.to_netcdf(save_path_help)
    print(save_path_help)

    ### SAVE MONTHLY VARIABLE TELECONNECTION STRENGTHS TO NETCDF
    psi_1 = xr.DataArray(corr_array1,
                       coords = {"month":  np.arange(1, int(corr_array1.shape[0])+1),
                                 "latitude":  ds1['latitude'],
                                 "longitude":  ds1['longitude']
                                 },
                        dims = ["month", "latitude", "longitude"],
                        attrs=dict(
                            variable=var1_str,
                            description=psi_str,
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = climate_index)
                            )
    
    save_path_help = save_path + "/psi_" + climate_index + "_" + var1_str + "_LAND_nskip{:.1f}".format(nskip) + "_" +\
        str(start_year) + str(end_year) + "_12months.nc"
    psi_1.to_netcdf(save_path_help)

    psi_2 = xr.DataArray(corr_array2,
                       coords = {"month":  np.arange(1, int(corr_array2.shape[0])+1),
                                 "latitude":  ds1['latitude'],
                                 "longitude":  ds1['longitude']
                                 },
                        dims = ["month", "latitude", "longitude"],
                        attrs=dict(
                            variable=var2_str,
                            description=psi_str,
                            psi_calc_start_date = str(datetime(start_year, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(end_year, 12, 1, 0, 0, 0)),
                            climate_index_used = climate_index)
                            )
    
    save_path_help = save_path + "/psi_" + climate_index + "_" + var2_str + "_LAND_nskip{:.1f}".format(nskip) + "_" +\
        str(start_year) + str(end_year) + "_12months.nc"
    psi_2.to_netcdf(save_path_help)
    
    ### PLOT TELECONNECTION
    if (plot_psi == True):
        path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
        gdf = gpd.read_file(path_land)

        fig, ax = plt.subplots(figsize=(10, 6))
        psi.plot(
            x="longitude", 
            y="latitude",
            ax=ax,
            cmap="YlOrRd"
        )
        gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)
        ax.set_title("Teleconnection with Country Outlines")
        plt.show()



#
compute_teleconnection(nc_path = '/Users/tylerbagwell/Downloads/data_stream-moda.nc',
                       save_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections',
                       nskip = 5,
                       climate_index = 'ani', 
                       start_year = 1950,
                       end_year = 2023,
                       plot_psi = True)