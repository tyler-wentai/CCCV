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
from oldcode.prepare_index import *

print('\n\nSTART ---------------------\n')

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


### ---- Initialize and prepare the geo data ---- ###

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




### ---- Initialize and prepare the conflict data ---- ###

conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'
conflict_df = pd.read_csv(conflictdata_path)

conflict_df['country_name'] = conflict_df['country'].replace(replacements) # this is used to make names of countries match between data sets
conflict_df.drop(['country'], axis=1, inplace=True)
conflict_df.rename(columns={'country_name': 'country'}, inplace=True)

# 1. Remove events that do not belong to years containing an active conflict dyad
conflict_df = conflict_df[conflict_df['type_of_violence']==3]
conflict_df = conflict_df[conflict_df['active_year']==1]

# 2. 
grouped = conflict_df.groupby(['dyad_new_id', 'year', 'country'])['best'].sum().reset_index()
grouped['onset'] = (grouped['best'] >= 25).astype(int)
grouped = grouped.sort_values(['dyad_new_id', 'country', 'year'])

years = range(start_year, (end_year+1))

# Get unique dyad_id and country combinations
dyad_country = grouped[['dyad_new_id', 'country']].drop_duplicates()


complete = dyad_country.assign(key=1).merge(
    pd.DataFrame({'year': years, 'key': 1}),
    on='key'
).drop('key', axis=1)

complete = complete.merge(
    grouped[['dyad_new_id', 'country', 'year', 'onset']],
    on=['dyad_new_id', 'country', 'year'],
    how='left'
)
complete['onset'] = complete['onset'].fillna(0).astype(int)
complete = complete.sort_values(['dyad_new_id', 'country', 'year'])

complete['onset_lag1y'] = complete.groupby(['dyad_new_id', 'country'])['onset'].shift(1)
complete = complete.dropna(subset=['onset_lag1y'])

# Set flag to 0 if current flag is 1 and previous flag is also 1
complete['onset'] = np.where(
    (complete['onset'] == 1) & (complete['onset_lag1y'] == 1),
    0,
    complete['onset']
)

onset_sum = complete.groupby(['country', 'year'])['onset'].sum().reset_index()  # total count of unique dyad onsets for each country-year
onset_sum['onset'] = (onset_sum['onset'] > 0).astype(int)                       # boolean of at least one dyad onset for each country-year



### add in the missing countries not in the conflict data set

countriesA = set(gdf1['SOVEREIGNT'])
countriesB = set(onset_sum['country'])

missing_countries = list(countriesA - countriesB)

missing_df = pd.MultiIndex.from_product(
    [missing_countries, years],
    names=['country', 'year']
).to_frame(index=False)

missing_df['onset'] = 0

# Append the missing_df to the existing flags_sum
onset_sum = pd.concat([onset_sum, missing_df], ignore_index=True)
panel = onset_sum.sort_values(['country', 'year']).reset_index(drop=True)






### add the climate index to the panel 
nlag_psi = 4

# annual_index = compute_annualized_NINO3_index((start_year-nlag_psi-1), end_year)
annual_index = compute_annualized_DMI_index((start_year-nlag_psi-1), end_year)

for i in range(nlag_psi+1):
    lag_string = 'INDEX_lag' + str(i) + 'y'
    annual_index[lag_string] = annual_index['INDEX'].shift((i))
lag_string = 'INDEX_lagF' + str(1) + 'y'                            # Add a forward lag
# annual_index[lag_string] = annual_index['INDEX'].shift((-1))        # Add a forward lag
annual_index.drop('INDEX', axis=1, inplace=True)

panel = panel.merge(annual_index, on='year', how='left')

# panel = panel[panel['year'] != 2023] # !!!!!!!!!!!!!!!!!!!!!!!!
panel = panel.reset_index(drop=True)



### add the teleconnection strength (psi) to the panel 
print('Computing gdf for psi...')

telecon_path = "/Users/tylerbagwell/Desktop/psi_callahan_DMI.nc"
psi = xr.open_dataarray(telecon_path)
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon']) ### REMOVE IF NOT USING psi_Hsiang2011_nino3.nc !!!!
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


fig, ax = plt.subplots(1, 1, figsize=(10, 6))
psi_geom.plot(
            column='psi',    
            cmap='bwr',   #turbo    YlOrRd
            legend=True,                   
            legend_kwds={'label': "psi", 'orientation': "horizontal"},           
            ax=ax,
        )
plt.title('Country aggregated teleconnection strength, Hsiang 2011 Method')
fig.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/Hsiang_2011_ENSO_Teleconnection_CountryAggregate.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

print(panel)

panel.to_csv('/Users/tylerbagwell/Desktop/Onset_global_dmi_Callahan_CON3.csv', index=False)