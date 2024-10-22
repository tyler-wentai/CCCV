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

print("\n")

replacements = {
                'Bosnia-Herzegovina':'Bosnia and Herzegovina',
                'Cambodia (Kampuchea)':'Cambodia',
                'DR Congo (Zaire)':'Democratic Republic of the Congo',
                'Kingdom of eSwatini (Swaziland)':'eSwatini',
                'Guinea-Bissau':'Guinea-Bissau',
                'Madagascar (Malagasy)':'Madagascar',
                'Myanmar (Burma)':'Myanmar',
                'Serbia (Yugoslavia)':'Republic of Serbia',
                'Congo':'Republic of the Congo',
                'Russia (Soviet Union)':'Russia',
                'Tanzania':'United Republic of Tanzania', 
                'Yemen (North Yemen)':'Yemen', 
                'Zimbabwe (Rhodesia)':'Zimbabwe'
                }


conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'
conflict_df = pd.read_csv(conflictdata_path)

# print(len(set(conflict_df['conflict_new_id'])))

# print(conflict_df.shape)
# print(conflict_df[conflict_df['best']>=1].shape)

print(conflict_df.shape)
conflict_df = conflict_df[conflict_df['best']>=1]
conflict_df = conflict_df[conflict_df['active_year']==1]
print(conflict_df.shape)


country_years = (
    conflict_df.groupby(['country', 'conflict_new_id'])['year']
      .min()
      .reset_index()
      .groupby('country')['year']
      .apply(lambda years: sorted(years.unique()))
      .reset_index(name='onset_years')
)

country_years['country_name'] = country_years['country'].replace(replacements) # this is used to make names of countries match between data sets
country_years.drop(['country'], axis=1, inplace=True)
country_years.rename(columns={'country_name': 'country'}, inplace=True)

print(country_years)
# print(country_years.iloc[0])



# 1. Define the range of years
start_year = 1989
end_year = 2023
years = list(range(start_year, end_year + 1))

# 2. Get the list of unique countries
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
gdf1 = gpd.read_file(path_land)
regions = set(gdf1['SOVEREIGNT'])
countries = regions


countriesA = set(regions)
countriesB = set(country_years['country'])

missing_countries = list(countriesA - countriesB)
df_missing  = pd.DataFrame({'country': missing_countries})
df_missing['onset_years'] = [[] for _ in range(len(df_missing))]


country_years = pd.concat([country_years, df_missing], ignore_index=True)
print(country_years)
# print(sorted(countries))

# 3. Create a Cartesian product of countries and years
panel = pd.MultiIndex.from_product(
    [countries, years],
    names=['country', 'year']
    ).to_frame(index=False)

# 4. Merge with 'country_years' to access 'onset_years'
panel = panel.merge(country_years, on='country', how='left')


# 5. Create 'conflict_onset' indicator
panel['conflict_onset'] = panel.apply(
    lambda row: 1 if row['year'] in row['onset_years'] else 0,
    axis=1
)

# 6. (Optional) Drop 'first_years' if no longer needed
panel = panel.drop('onset_years', axis=1)

# 7. (Optional) Sort the panel for readability
panel = panel.sort_values(['country', 'year']).reset_index(drop=True)




