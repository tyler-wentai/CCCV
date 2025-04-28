import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from build_panel_state import initalize_state_onset_panel

print('\n\nSTART ---------------------\n')

var1_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_t2m_raw.nc'
var2_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc'
resolution = 0.25


panel = initalize_state_onset_panel(panel_start_year=1950,
                                    panel_end_year=2023,
                                    telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_ani_res0.25_19502023_pval0.05.nc',
                                    pop_path = '/Users/tylerbagwell/Desktop/cccv_data/gpw-v4-population-count-rev11_totpop_15_min_nc/gpw_v4_population_count_rev11_15_min.nc',
                                    clim_index='ani',
                                    plot_telecon=True)


##
ds1 = xr.open_dataset(var1_path)
var1_str = list(ds1.data_vars)[0]

ds2 = xr.open_dataset(var2_path)
var2_str = list(ds2.data_vars)[0]

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

###
import rioxarray
from rasterstats import zonal_stats
da_yearly = ds1.t2m.groupby("valid_time.year").mean("valid_time")
# da_yearly = ds2.tp.groupby("valid_time.year").mean("valid_time")

# Ensure the DataArray has the proper CRS (if not already set)
da_yearly.rio.write_crs("EPSG:4326", inplace=True)

results = []

# 2. Group your geodataframe by year.
for year, group in panel.groupby("year"):
    # 3. Select the corresponding yearly slice from the xarray.
    da = da_yearly.sel(year=year)
    
    # Loop over each polygon (each unique fid) in the group.
    for _, row in group.iterrows():
        # 4. Compute the spatial (zonal) mean using the polygon's geometry.
        stats = zonal_stats(
            row.geometry, 
            da.values, 
            affine=da.rio.transform(),  # Get the transformation info
            stats='mean'
        )
        spatial_mean = stats[0]['mean']
        
        results.append({
            'year': year,
            'fid': row.fid,
            'spatial_avg': spatial_mean
        })

# Convert results to a DataFrame for further analysis.
result_df = pd.DataFrame(results)

panel = panel.merge(result_df, on=['year', 'fid'], how='left')
print(panel)

panel = panel.drop('geometry', axis=1)
panel.to_csv('/Users/tylerbagwell/Desktop/panel_datasets/ag_panels/validation_t2m_ani.csv', index=False)