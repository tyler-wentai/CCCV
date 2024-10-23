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
from prepare_index import *

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


start_year = 1989
end_year = 2023
years = list(range(start_year, end_year + 1))

remove_sovereignty = ['Antigua and Barbuda', 'Federated States of Micronesia', 'Kiribati', 'Nauru', 'Marshall Islands', 'Maldives', 
                        'Palau', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Seychelles',
                        'Tuvalu', 'Vanuatu', 'Mauritius', 'São Tomé and Principe', 'Cabo Verde', 'Antarctica',
                        'Vatican','Tonga','Singapore','Monaco','Malta','Liechtenstein','Grenada','Dominica','Barbados',
                        'Bahrain','Andorra']

# read in shp file data
path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
gdf1 = gpd.read_file(path_land)
regions = set(gdf1['SOVEREIGNT'])
gdf1 = gdf1[gdf1['SOVEREIGNT'].isin(regions)]
gdf1 = gdf1[gdf1['SOVEREIGNT'] == gdf1['ADMIN']]
gdf1 = gdf1[~gdf1['SOVEREIGNT'].isin(remove_sovereignty)]

# print(sorted(gdf1['SOVEREIGNT']))


# print(gdf1)




conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'
conflict_df = pd.read_csv(conflictdata_path)

# print(conflict_df.shape)
conflict_df = conflict_df[conflict_df['best']>=1]
conflict_df = conflict_df[conflict_df['active_year']==1]
# print(conflict_df.shape)

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

# print(country_years)
# print(country_years.iloc[0])



countriesA = set(gdf1['SOVEREIGNT'])
countriesB = set(country_years['country'])

missing_countries = list(countriesA - countriesB)
df_missing  = pd.DataFrame({'country': missing_countries})
df_missing['onset_years'] = [[] for _ in range(len(df_missing))]


country_years = pd.concat([country_years, df_missing], ignore_index=True)
# print(country_years)
# print(sorted(countries))

# 3. Create a Cartesian product of countries and years
panel = pd.MultiIndex.from_product(
    [countriesA, years],
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





nlag_psi = 4

annual_index = compute_annualized_NINO3_index((start_year-nlag_psi-1), end_year)
#annual_index = compute_annualized_DMI_index(start_year, end_year)

for i in range(nlag_psi+1):
    lag_string = 'INDEX_lag' + str(i) + 'y'
    annual_index[lag_string] = annual_index['INDEX'].shift((i))
lag_string = 'INDEX_lagF' + str(1) + 'y'                            # Add a forward lag
annual_index[lag_string] = annual_index['INDEX'].shift((-1))        # Add a forward lag
annual_index.drop('INDEX', axis=1, inplace=True)

panel = panel.merge(annual_index, on='year', how='left')

panel = panel[panel['year'] != 2023]
panel = panel.reset_index(drop=True)
print(panel)


# print(panel)


print('Computing gdf for psi...')
telecon_path = "/Users/tylerbagwell/Desktop/psi_callahan_NINO3_0dot5_soilw.nc"
psi = xr.open_dataarray(telecon_path)
if 'lat' not in psi.coords or 'lon' not in psi.coords:
    raise ValueError("DataArray must have 'lat' and 'lon' coordinates.")

df_psi = psi.to_dataframe(name='psi').reset_index()
df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['lon'], row['lat']), axis=1)
psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
psi_gdf = psi_gdf[['lat', 'lon', 'psi', 'geometry']]

if psi_gdf.crs != gdf1.crs:
    psi_gdf = psi_gdf.to_crs(gdf1.crs)
    print("Reprojected gdf to match final_gdf CRS.")

joined_gdf = gpd.sjoin(psi_gdf, gdf1, how='left', predicate='within')

cleaned_gdf = joined_gdf.dropna(subset=['SOVEREIGNT'])
cleaned_gdf = cleaned_gdf.reset_index(drop=True)

grouped = joined_gdf.groupby('SOVEREIGNT')
mean_psi = grouped['psi'].mean().reset_index() # Computing aggregated psi using the MAX of all psis in polygon

# for randomizing psi:
# mean_psi['psi'] = np.random.permutation(mean_psi['psi']) # MAKE SURE TO COMMENT OUT!!!!!

psi_geom = mean_psi.merge(gdf1, on='SOVEREIGNT', how='left')
psi_geom = gpd.GeoDataFrame(psi_geom, geometry='geometry')
mean_psi.rename(columns={'SOVEREIGNT': 'country'}, inplace=True)

panel = panel.merge(mean_psi, on='country', how='left')

# print(panel)
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
psi_geom.plot(
            column='psi',    
            cmap='YlOrRd',   #turbo    YlOrRd
            legend=True,                   
            legend_kwds={'label': "psi", 'orientation': "horizontal"},           
            ax=ax
        )
plt.show()


panel.to_csv('/Users/tylerbagwell/Desktop/Onset_global_nino3.csv', index=False)
