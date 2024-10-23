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

print('\n\nSTART ---------------------\n')

conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'
conflict_df = pd.read_csv(conflictdata_path)

# 1. Remove events that do not belong to years containing an active conflict dyad
conflict_df = conflict_df[conflict_df['active_year']==1]

# 2. 
grouped = conflict_df.groupby(['dyad_new_id', 'year', 'country'])['best'].sum().reset_index()
grouped['onset'] = (grouped['best'] >= 25).astype(int)

grouped = grouped.sort_values(['dyad_new_id', 'country', 'year'])

years = range(1989, 2024)

# Get unique dyad_id and country combinations
dyad_country = grouped[['dyad_new_id', 'country']].drop_duplicates()

print(dyad_country)

complete = dyad_country.assign(key=1).merge(
    pd.DataFrame({'year': years, 'key': 1}),
    on='key'
).drop('key', axis=1)

print(complete)