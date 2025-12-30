import numpy as np
from scipy.stats import linregress
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
from pingouin import partial_corr
from utils.calc_annual_index import *
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

print('\n\nSTART ---------------------\n')
# COMPUTES CUMULATIVE CORRELATIONS FOR A SINGLE GRIDDED VARIABLE AND A SINGLE CLIMATE INDEX USING THE CAI ET AL. 2024 METHOD

# --- USER PARAMETERS --- #
clim_index = 'NINO3'
start_year  = 1950
end_year    = 2024 
file_path_VAR1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/raw_climate_data/mrsos_ERA5_mon_194001-202412_v02.nc'
var1str = 'mrsos' 
resolution = 1.0
# ------------------------ #

ds1 = xr.open_dataset(file_path_VAR1)

# ds1 = ds1.rename({'valid_time': 'time'})
ds1 = ds1.assign_coords(time=ds1.time.dt.floor('D'))

# change dates to time format:
ds1 = ds1.rename({'lat': 'latitude'})
ds1 = ds1.rename({'lon': 'longitude'})

# Access longitude and latitude coordinates
lon1 = ds1['longitude']
lat1 = ds1['latitude']

lat_int_mask = (lat1 % resolution == 0)
lon_int_mask = (lon1 % resolution == 0)
ds1 = ds1.sel(latitude=lat1[lat_int_mask], longitude=lon1[lon_int_mask])


# nskip = 20
# ds1 = ds1.isel(latitude=slice(0, None, int(nskip)), longitude=slice(0, None, int(nskip)))


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

ds1 = ds1.assign_coords(
    longitude=np.round(ds1['longitude'], decimals=2),
    latitude=np.round(ds1['latitude'], decimals=2)
)

if clim_index == 'NINO3':
    print("Preparing NINO3 index...")
    clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
                        end_date=datetime(end_year, 12, 1, 0, 0, 0))
elif clim_index == 'NINO34':
    print("Preparing NINO34 index...")
    clim_ind = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
                        start_date=datetime(start_year, 1, 1, 0, 0, 0),
                        end_date=datetime(end_year, 12, 1, 0, 0, 0))
elif clim_index == 'DMI':
    print("Preparing DMI index...")
    clim_ind = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
else:
    print("Error: clim_index not recognized. Please use 'NINO3', 'NINO34', or 'DMI'.")
    sys.exit(1)

# load index data
# clim_ind = prepare_Eindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))
# clim_ind = prepare_Cindex(file_path='data/CE_index.csv',
#                         start_date=datetime(start_year, 1, 1, 0, 0, 0),
#                         end_date=datetime(end_year, 12, 1, 0, 0, 0))

common_lon  = np.intersect1d(ds1['longitude'], ds1['longitude'])
common_lat  = np.intersect1d(ds1['latitude'], ds1['latitude'])
common_time = np.intersect1d(ds1['time'], clim_ind.index.to_numpy())

ds1_common      = ds1.sel(time=common_time, longitude=ds1['longitude'], latitude=ds1['latitude'])
clim_ind_common = clim_ind.loc[clim_ind.index.isin(pd.to_datetime(common_time))]

var1_common = ds1_common[var1str]

# Check shapes
t = np.asarray(common_time).astype("datetime64[ns]")
min_year_final = pd.Timestamp(t.min()).year
max_year_final = pd.Timestamp(t.max()).year
print("var1_common shape:", var1_common.shape)
print("clim_ind shape:   ", clim_ind_common.shape)

n_time, n_lat, n_long = var1_common.shape

# Verify that coordinates are identical
assert np.array_equal(var1_common['time'], clim_ind_common.index)


def detrend_monthly(data):
    """
    1. Remove the mean seasonal cycle (12-month climatology)
    2. Detrend each calendar-month slice separately
    """
    # set‑up 
    data   = np.asarray(data, dtype=float)
    n      = data.size
    months = np.arange(n) % 12            # 0...11

    #  1. climatology 
    clim_mean = np.array([data[months == m].mean() for m in range(12)])
    anom      = data - clim_mean[months]  # mean removed

    # 2. detrend each calendar month 
    df = pd.DataFrame({
        "anom" : anom,
        "month": months,
        "time" : np.arange(n, dtype=float)
    })

    def _remove_trend(group):
        if len(group) < 2: # safety for very short series
            return group["anom"]
        slope, intercept, *_ = linregress(group["time"], group["anom"])
        return group["anom"] - (slope * group["time"] + intercept)

    df["detr"] = (
        df.groupby("month", group_keys=False)
            .apply(_remove_trend, include_groups=False))

    return df["detr"].values.tolist()


anom_file1 = Path('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_climate_data/mrsos_anom_nostd_ERA5_1d0_19502024_FINAL.npy')

if anom_file1.exists():
    print("Anomaly field file exists. Skipping processing.")

    var1_std = np.load(anom_file1)
    # shape
    print(var1_std.shape)

else:
    print("Anomaly field file is missing. Proceeding with processing.")
    var1_std = np.empty_like(var1_common) # Initialize a new array to store the standardized data

    print("Standardizing and de-trending climate variable data...")
    for i in range(n_lat):
        if (i%10==0): 
            print("...", i)
        for j in range(n_long):
            var1_std[:, i, j] = detrend_monthly(var1_common[:, i, j])

    np.save('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_climate_data/mrsos_anom_nostd_ERA5_1d0_' + str(min_year_final) + str(max_year_final) + '_FINAL.npy', var1_std)


# Compute the annualized index value:
clim_ind_common.index = pd.to_datetime(clim_ind_common.index)     # Ensure 'date' to datetime and extract year & month
clim_ind_common = clim_ind_common.copy()
clim_ind_common['year'] = clim_ind_common.index.year
clim_ind_common['month'] = clim_ind_common.index.month

## --- NINO3 or NINO34
if clim_index == 'NINO3' or clim_index == 'NINO34':
    print("Calculating NINO3 / NINO3.4 average index for NDJ...")

    n_months = 12   # ENSO "year": May_t to Apr_t+1

    jan_df = clim_ind_common[clim_ind_common['month'] == 1].copy() # prepare January data from following year
    jan_df['year'] = jan_df['year'] - 1  # Shift back a year
    jan_df = jan_df[['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})

    nov_dec_df = clim_ind_common[clim_ind_common['month'].isin([11, 12])].copy() # prepare November and December data for current year
    nov     = nov_dec_df[nov_dec_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})
    dec     = nov_dec_df[nov_dec_df['month'] == 12][['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    yearly = pd.merge(jan_df, nov, on='year', how='inner') # merge November_t, January_t+1 data
    yearly = pd.merge(yearly, dec, on='year', how='inner') # merge November_t, December_t, January_t+1 data

    yearly['anom_ANN'] = yearly[['NOV_ANOM', 'DEC_ANOM', 'JAN_ANOM']].mean(axis=1) # Calculate the average NDJ ANOM value
    index_ANN = yearly[['year', 'anom_ANN']].sort_values('year').reset_index(drop=True)

    index_ANN["phase"] = np.where(index_ANN["anom_ANN"] >= 0.5, "pos",
                                  np.where(index_ANN["anom_ANN"] <= -0.5, "neg", "neu"))
    
    print(index_ANN)
    phase_counts = index_ANN["phase"].value_counts(dropna=False)
    print(phase_counts)
## --- DMI
elif clim_index == 'DMI':
    print("Calculating DMI average index for SON...")

    n_months = 8    # IOD "year": May_t to Dec_t

    sep_oct_nov_df = clim_ind_common[clim_ind_common['month'].isin([9, 10, 11])].copy() # prepare January and February data for current year
    sep     = sep_oct_nov_df[sep_oct_nov_df['month'] == 9][['year', 'ANOM']].rename(columns={'ANOM': 'SEP_ANOM'})
    oct     = sep_oct_nov_df[sep_oct_nov_df['month'] == 10][['year', 'ANOM']].rename(columns={'ANOM': 'OCT_ANOM'})
    nov     = sep_oct_nov_df[sep_oct_nov_df['month'] == 11][['year', 'ANOM']].rename(columns={'ANOM': 'NOV_ANOM'})

    yearly = pd.merge(sep, oct, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, nov, on='year', how='inner') # merge December, January, and February data

    yearly['anom_ANN'] = yearly[['SEP_ANOM', 'OCT_ANOM', 'NOV_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_ANN = yearly[['year', 'anom_ANN']].sort_values('year').reset_index(drop=True)

    index_ANN["phase"] = np.where(index_ANN["anom_ANN"] >= 0.4, "pos",
                                  np.where(index_ANN["anom_ANN"] <= -0.4, "neg", "neu"))
    
    print(index_ANN)
    phase_counts = index_ANN["phase"].value_counts(dropna=False)
    print(phase_counts)
else:
    print("Error: clim_index not recognized. Please use 'NINO3', 'NINO34', or 'DMI'.")
    sys.exit(1)


############################
pos_phase_psi = np.empty((2, n_lat,n_long))
neg_phase_psi = np.empty((2, n_lat,n_long))

def compute_d_stats(df_in, clim_index):
    # compute d statistics column-wise

    var_allyears = df_in.drop(columns=['year', 'phase', 'anom_ANN'])
    var_posyears = df_in[df_in['phase'] == 'pos'].drop(columns=['year', 'phase', 'anom_ANN']).reset_index(drop=True)
    var_negyears = df_in[df_in['phase'] == 'neg'].drop(columns=['year', 'phase', 'anom_ANN']).reset_index(drop=True)

    avg_allyears = var_allyears.mean(axis=0)
    avg_posyears = var_posyears.mean(axis=0)
    avg_negyears = var_negyears.mean(axis=0)

    std_allyears = var_allyears.std(axis=0)
    std_posyears = var_posyears.std(axis=0)
    std_negyears = var_negyears.std(axis=0)

    n_allyears = var_allyears.shape[0]
    n_posyears = var_posyears.shape[0]
    n_negyears = var_negyears.shape[0]

    # compute d statistic for pos - all
    d_pos = (avg_posyears - avg_allyears) * np.sqrt(n_posyears + n_allyears - 2)
    d_pos /= np.sqrt( (n_posyears - 1)*std_posyears**2 + (n_allyears - 1)*std_allyears**2 )

    # compute d statistic for neg - all
    d_neg = (avg_negyears - avg_allyears) * np.sqrt(n_negyears + n_allyears - 2)
    d_neg /= np.sqrt( (n_negyears - 1)*std_negyears**2 + (n_allyears - 1)*std_allyears**2 )

    # compute 3-point rolling averages
    if (clim_index == 'NINO3' or clim_index == 'NINO34'):
        # wrap around for Jan and Dec
        d_pos_vals = d_pos.to_numpy()
        wrap_pos = (np.roll(d_pos_vals, 1) + d_pos_vals + np.roll(d_pos_vals, -1)) / 3
        d_pos_roll3 = pd.Series(wrap_pos, index=d_pos.index)

        d_neg_vals = d_neg.to_numpy()
        wrap_neg = (np.roll(d_neg_vals, 1) + d_neg_vals + np.roll(d_neg_vals, -1)) / 3
        d_neg_roll3 = pd.Series(wrap_neg, index=d_neg.index)
    elif (clim_index == 'DMI'): ### CHECK THIS LATER!!!!!
        d_pos_roll3 = d_pos.rolling(window=3, center=True, min_periods=1).mean()
        d_neg_roll3 = d_neg.rolling(window=3, center=True, min_periods=1).mean()
    else:
        print("Error: clim_index not recognized.")
        sys.exit(1)

    # grab maximum absolute d statistic and corresponding month
    month_max_mag_pos = d_pos_roll3.abs().idxmax()
    value_max_mag_pos = d_pos_roll3.loc[month_max_mag_pos]
    month_max_mag_pos_int = int(month_max_mag_pos.split('_')[-1])

    month_max_mag_neg = d_neg_roll3.abs().idxmax()
    value_max_mag_neg = d_neg_roll3.loc[month_max_mag_neg]
    month_max_mag_neg_int = int(month_max_mag_neg.split('_')[-1])

    return value_max_mag_pos, month_max_mag_pos_int, value_max_mag_neg, month_max_mag_neg_int


def label_wet_dry_none(df_in, clim_index, alpha=0.05):
    # compute true d statistics
    d_pos, d_pos_month, d_neg, d_neg_month = compute_d_stats(df_in, clim_index)

    # bootstrap to compute null distribution of d statistics
    m = 100
    d_pos_bootstrap = []
    d_neg_bootstrap = []
    for _boot in range(m):
        df_shuffled = df_in.copy()
        df_shuffled['phase'] = np.random.permutation(df_shuffled['phase'].values) # WILL MAKE THIS THREE-YEAR BLOCKED LATER!!!

        d_pos_b, _, d_neg_b, _ = compute_d_stats(df_shuffled, clim_index)

        d_pos_bootstrap.append(d_pos_b)
        d_neg_bootstrap.append(d_neg_b)

    # plt.hist(d_pos_bootstrap, bins='scott', alpha=0.5, color='blue', density=True)
    # plt.hist(d_neg_bootstrap, bins='scott', alpha=0.5, color='red', density=True)
    # plt.axvline(d_pos, color='blue', linestyle='dashed', linewidth=2, label='Observed d (pos)')
    # plt.axvline(d_neg, color='red', linestyle='dashed', linewidth=2, label='Observed d (neg)')
    # plt.text(-0.8, 0.95*plt.ylim()[1], f'd_pos = {d_pos:.2f}', color='blue')
    # plt.text(-0.8, 0.90*plt.ylim()[1], f'd_neg = {d_neg:.2f}', color='red')
    # plt.xlim(-1.0, 1.0)
    # plt.show()

    # compute p-values
    p_val_pos = (np.sum(np.abs(d_pos_bootstrap) >= np.abs(d_pos)) + 1) / (m + 1)
    p_val_neg = (np.sum(np.abs(d_neg_bootstrap) >= np.abs(d_neg)) + 1) / (m + 1)

    if p_val_pos < alpha:
        d_val_pos = d_pos
    else:
        d_val_pos = 0.
        d_pos_month = -1

    if p_val_neg < alpha:
        d_val_neg = d_neg
    else:
        d_val_neg = 0.
        d_neg_month = -1

    return d_val_pos, d_pos_month, d_val_neg, d_neg_month



print("\nComputing psi array...")
warnings.filterwarnings('ignore', category=RuntimeWarning) #!!!!!!!
for i in range(n_lat):
    if (i%10==0): 
        print("...", i)
    for j in range(n_long):
        if (j%30==0): 
            print("...", i, j)
        current_vars = pd.DataFrame(data=var1_std[:,i,j],
                                            index=var1_common['time'], #need to use var1_common since it still contains the time data
                                            columns=[var1str])
        current_vars.index = pd.to_datetime(current_vars.index)
        current_vars['year'] = current_vars.index.year
        current_vars['month'] = current_vars.index.month
        
        # pivot to wide format
        wide = (current_vars
            .pivot(index="year", columns="month", values="mrsos")
            .reindex(columns=range(1, 13)))  # enforce months 1..12 order
        wide.columns = [f"{var1str}_{m}" for m in wide.columns]
        arr = np.column_stack([wide.index.to_numpy(), wide.to_numpy()])
        current_vars = wide.reset_index()


        if clim_index == 'NINO3' or clim_index == 'NINO34':
            cols_up = [f"{var1str}_{m}" for m in range(1, 5)]
            current_vars[cols_up] = current_vars[cols_up].shift(-1) # shift Jan-Apr up by 1 year to match ENSO year
            current_vars = current_vars.iloc[:-1].reset_index(drop=True)
            enso_cols = [f"{var1str}_{m}" for m in range(5, 13)] + [f"{var1str}_{m}" for m in range(1, 5)] # May-Dec + Jan-Apr (makes it easier when we do the 3-month rolling avg.)
            current_vars = current_vars[["year"] + enso_cols]
        elif clim_index == 'DMI':
            cols_up = [f"{var1str}_{m}" for m in range(5, 13)]
            current_vars = current_vars[cols_up] # only keep May-Dec for IOD year
        else:
            print("Error: clim_index not recognized. Please use 'NINO3', 'NINO34', or 'DMI'.")
            sys.exit(1)

        
        has_nan = current_vars.isna().to_numpy().any()
        if has_nan==False:
            current_vars = pd.merge(current_vars, index_ANN, how='inner', on='year')
            d_val_pos, d_pos_month, d_val_neg, d_neg_month = label_wet_dry_none(current_vars, clim_index, alpha=0.05)
        else:
            d_val_pos, d_pos_month, d_val_neg, d_neg_month = np.nan, np.nan, np.nan, np.nan

        pos_phase_psi[0,i,j] = d_val_pos
        pos_phase_psi[1,i,j] = d_pos_month
        neg_phase_psi[0,i,j] = d_val_neg
        neg_phase_psi[1,i,j] = d_neg_month

# Positive phase
pos_psi_array = xr.DataArray(data = pos_phase_psi,
                            coords={"var": ["psi_pos", "psi_pos_month"],
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["var", "lat", "lon"],
                        attrs=dict(
                            description=f"Non-parametric wet/dry {var1str} teleconnections during positive phase of {clim_index}.",
                            psi_calc_start_date = str(datetime(min_year_final, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(max_year_final, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathA_str = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_' + var1str + "_pos-" + clim_index +'.nc'
pos_psi_array.to_netcdf(pathA_str)

# Negative phase
neg_psi_array = xr.DataArray(data = neg_phase_psi,
                            coords={"var": ["psi_neg", "psi_neg_month"],
                            "lat": common_lat,
                            "lon": common_lon
                        },
                        dims = ["var", "lat", "lon"],
                        attrs=dict(
                            description=f"Non-parametric wet/dry {var1str} teleconnections during negative phase of {clim_index}.",
                            psi_calc_start_date = str(datetime(min_year_final, 1, 1, 0, 0, 0)),
                            psi_calc_end_date = str(datetime(max_year_final, 12, 1, 0, 0, 0)),
                            climate_index_used = clim_index,
                            resolution = resolution)
                        )

pathB_str = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/processed_teleconnections/psi_' + var1str + "_neg-" + clim_index +'.nc'
neg_psi_array.to_netcdf(pathB_str)