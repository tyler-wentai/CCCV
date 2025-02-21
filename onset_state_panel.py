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

# read in country border data
shape_path = "data/cshape_files/cshpR.shp"
df = gpd.read_file(shape_path)

# need to compute the weighted population grid for each country
# import rasterio

#df = df[df['cntry_n']=='Afghanistan']



# Convert the 'start' and 'end' columns to datetime and extract the year
df['start'] = pd.to_datetime(df['start'])
df['end'] = pd.to_datetime(df['end'])
df['year'] = df['start'].dt.year

# Create a panel with country-year observations
panel_list = []


## NEED TO CHECK IF THERE ARE MULTIPLE CHANGES FOR A COUNTRY IN THE SAME YEAR

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

# Concatenate the individual country panels
panel_df = pd.concat(panel_list, ignore_index=True)


# # Remove country-years observations before start_year
panel_df = panel_df[panel_df['year'] >= panel_start_year]


# # panel_df now contains one observation per country-year
# # print(panel_df[['cntry_n', 'year', 'gwcode', 'geometry']])
# print(panel_df[panel_df['cntry_n']=='Czechoslovakia'])
print(df[df['cntry_n']=='Russia (Soviet Union)'])
print(panel_df[panel_df['cntry_n']=='Russia (Soviet Union)'])