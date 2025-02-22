import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from datetime import datetime
import math
import xarray as xr
import seaborn as sns
from shapely.geometry import mapping
import statsmodels.api as sm
from prepare_index import *

print('\n\nSTART ---------------------\n')


### user params:
panel_start_year = 1950
panel_end_year = 2023

######## A. INITIALIZE PANEL
# read in country border data
shape_path = "data/cshape_files/cshpR.shp"
df = gpd.read_file(shape_path)

# Convert the 'start' and 'end' columns to datetime and extract the year
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df['year'] = df['start'].dt.year

# Create a panel with country-year observations
panel_list = []

# pre-processing before creation of panel:
# (1) if multiple border changes occur in a single year, pick the last one in that year
df = df.sort_values('start', ascending=True).drop_duplicates(subset=['cntry_n', 'year'], keep='last')
# (2) if the border change occurs after jun 31st in that year, change the border year to the following year
# NOT IMPLEMENTING (2)

dataset_max_year = df['end'].dt.year.max() # the last observed year in the cshape dataset

# Group by country
for country, group in df.groupby('cntry_n'):

    # Determine the panel year range
    start_year = group['year'].min()
    end_year = group['end'].dt.year.max()

    if (end_year >= dataset_max_year): # case where we extend 2019's (dataset_max_year) geometry to panel_end_year (we assume no major changes)
        end_year = panel_end_year
    elif (end_year < dataset_max_year): # case where the country's existence has ended before dataset_max_year
        end_year = end_year

    # Create a DataFrame with one row per year in the panel
    years = pd.DataFrame({'year': range(start_year, end_year + 1)})
    years['cntry_n'] = country

    # Sort both DataFrames by year
    group_sorted = group.sort_values('year')
    years_sorted = years.sort_values('year')
    years_sorted['year'] = years_sorted['year'].astype('int64')
    group_sorted['year'] = group_sorted['year'].astype('int64')
    
    # Merge using merge_asof to carry forward the last available geometry
    merged = pd.merge_asof(
        years_sorted,
        group_sorted,
        on='year',
        by='cntry_n',
        direction='backward'
    )
    
    panel_list.append(merged)

# combine the individual country panels
panel_df = pd.concat(panel_list, ignore_index=True)

# remove country-years observations before panel_start_year
panel_df = panel_df[panel_df['year'] >= panel_start_year]
panel_gdf = gpd.GeoDataFrame(panel_df, geometry='geometry')
panel_gdf.crs = "EPSG:4326"
panel_gdf = panel_gdf[['year','cntry_n','gwcode','fid','geometry']]


######## B. COMPUTE COUNTRY-YEAR TELECONNECTION STRENGTH
telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_ani_res0.5_19502023_pv0.01_ensoremovedv3.nc'

print('Computing gdf for psi...')
psi = xr.open_dataarray(telecon_path)

# Ensure that the DataArray has 'lat' and 'longitude' coordinates
if 'latitude' not in psi.coords or 'longitude' not in psi.coords:
    raise ValueError("DataArray must have 'latitude' and 'longitude' coordinates.")

df_psi = psi.to_dataframe(name='psi').reset_index()
df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['longitude'], row['latitude']), axis=1)
psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
psi_gdf = psi_gdf[['latitude', 'longitude', 'psi', 'geometry']]

# check crs
if psi_gdf.crs != panel_gdf.crs:
    psi_gdf = psi_gdf.to_crs(panel_gdf.crs)
    print("Reprojected gdf to match final_gdf CRS.")

unique_fids = panel_gdf[['fid', 'geometry']].drop_duplicates(subset='fid') # the fids are unique to each geometry in df and panel_gdf

joined_gdf = gpd.sjoin(psi_gdf, unique_fids, how='left', predicate='within')
joined_gdf = joined_gdf.dropna(subset=['fid'])
joined_gdf = joined_gdf.reset_index(drop=True)

grouped = joined_gdf.groupby('fid')
avg_psi = grouped['psi'].mean().reset_index() # Computing aggregated psi using the mean of all psi values within each country's geometry


panel_gdf = panel_gdf.merge(avg_psi, on='fid', how='left')
# print(panel_gdf[panel_gdf['cntry_n'] == 'Russia (Soviet Union)'])

last_obs = panel_gdf.loc[panel_gdf.groupby('cntry_n')['year'].idxmax()]

# Plot the geometries, coloring them by the psi value.
ax = last_obs.plot(column='psi', cmap='Reds', legend=True, figsize=(10, 6), edgecolor='black', linewidth=0.25)
plt.title("Psi")
plt.show()