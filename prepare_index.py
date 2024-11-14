
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

    # may_to_dec_df = clim_ind[clim_ind['month'].isin([5, 6, 7, 8, 9, 10, 11, 12])].copy() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
    # index_DJF = may_to_dec_df.groupby('year')['ANOM'].mean().reset_index() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
    # index_yrAVG = index_DJF.rename(columns={'ANOM': 'INDEX'}) # DELETE !!!!!!!!!!!!!!!!!!!!!!!!

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

    # may_to_dec_df = clim_ind[clim_ind['month'].isin([5, 6, 7, 8, 9, 10, 11, 12])].copy() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
    # index_DJF = may_to_dec_df.groupby('year')['ANOM'].mean().reset_index() # DELETE !!!!!!!!!!!!!!!!!!!!!!!!
    # index_yrAVG = index_DJF.rename(columns={'ANOM': 'INDEX'}) # DELETE !!!!!!!!!!!!!!!!!!!!!!!!

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



# start_year = 1989
# end_year = 2023

# dat = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt', 
#                     start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                     end_date=datetime(end_year, 12, 1, 0, 0, 0))

# index_yr = compute_annualized_NINO3_index(start_year,end_year)
# index_yr = index_yr.set_index('year')

# dat = prepare_DMI(file_path='data/NOAA_DMI_data.txt',
#                   start_date=datetime(1960, 1, 1, 0, 0, 0),
#                   end_date=datetime(2023, 12, 1, 0, 0, 0))

# index_yr = compute_annualized_DMI_index(1960,2023)
# index_yr = index_yr.set_index('year')

# monthly_avg = dat.groupby(dat.index.month)['ANOM'].mean()

# df_filtered_nino = dat[dat['ANOM'] >= 0]
# monthly_avg_nino = df_filtered_nino.groupby(df_filtered_nino.index.month)['ANOM'].mean()

# df_filtered_nina = dat[dat['ANOM'] <= 0]
# monthly_avg_nina = df_filtered_nina.groupby(df_filtered_nina.index.month)['ANOM'].mean()
# monthly_avg_nina = np.abs(monthly_avg_nina)


# plt.plot(np.arange(1,13), monthly_avg_nino, marker='o', color='red', label='nino3, El Nino')
# plt.plot(np.arange(1,13), monthly_avg_nina, marker='o', color='blue', label='nino3, La Nina (abs. val.)')
# plt.plot(np.arange(1,13), monthly_avg, marker='o', color='green', label='nino3')
# plt.xticks(range(1, 13))
# # Optionally, label the ticks with month names
# month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# plt.gca().set_xticklabels(month_names)
# plt.legend()
# plt.grid()
# plt.xlabel('Month')
# plt.ylabel('nino3 index averaged (degC)')
# plt.title('1990-2022')
# plt.savefig('/Users/tylerbagwell/Desktop/nino3_index_avg_1990.png', dpi=300, bbox_inches='tight')
# plt.show()


###

# dat['date'] = pd.to_datetime(dat.index)
# dat['Year'] = dat['date'].dt.year
# dat['Month'] = dat['date'].dt.month

# pivot_df = dat.pivot(index='Year', columns='Month', values='ANOM')
# pivot_df[[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]] = pivot_df[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].shift(-1)

# pivot_df['index_yravg'] = pivot_df.index.map(index_yr['INDEX'])
# pivot_df = pivot_df.dropna()

# print(pivot_df)



# from scipy.stats import pearsonr
# def pearson_corr_pvals(df):
#     cols = df.columns
#     corr = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
#     pvals = pd.DataFrame(np.zeros((len(cols), len(cols))), columns=cols, index=cols)
    
#     for i in range(len(cols)):
#         for j in range(len(cols)):
#             if i == j:
#                 corr.iloc[i, j] = 1.0
#                 pvals.iloc[i, j] = 0.0
#             elif i < j:
#                 rho, pval = pearsonr(df.iloc[:, i], df.iloc[:, j])
#                 corr.iloc[i, j] = rho
#                 pvals.iloc[i, j] = pval
#                 corr.iloc[j, i] = rho
#                 pvals.iloc[j, i] = pval
#     return corr, pvals

# corr_matrix, pval_matrix = pearson_corr_pvals(pivot_df)

# # print(corr_matrix)


# plt.plot(np.arange(1,25), corr_matrix['index_yravg'][0:24], color='red', label='corr', marker='.')
# plt.plot(np.arange(1,25), pval_matrix['index_yravg'][0:24], color='red', linestyle='--', label='p-value', marker='.')
# plt.xticks(range(1, 25))
# month_names = ['J', 'F', 'M', 'A', 'M', 'J',
#                'J', 'A', 'S', 'O', 'N', 'D',
#                'J', 'F', 'M', 'A', 'M', 'J',
#                'J', 'A', 'S', 'O', 'N', 'D']
# plt.gca().set_xticklabels(month_names)
# rectangle = plt.Rectangle((10.5, -1), 3, 3, linewidth=1, edgecolor=None, facecolor='gray', alpha=0.5)
# plt.gca().add_patch(rectangle)
# plt.xlim(0.75,24.25)
# plt.legend()
# plt.title("NINO3, 1989-2023")
# plt.ylabel("Correlation with NDJ NINO3")
# # plt.savefig('/Users/tylerbagwell/Desktop/nino3_NDJ_correlation.png', dpi=300, bbox_inches='tight')
# plt.show()