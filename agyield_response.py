import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from onset_state_panel import initalize_state_onset_panel

print('\n\nSTART ---------------------\n')

ag_path = '/Users/tylerbagwell/Desktop/cccv_data/ag_data/UNNA_ValuedAddedByEconomicActivity.csv'
ag_df = pd.read_csv(ag_path)


mapping_dict = {
    'Kyrgyzstan': 'Kyrgyz Republic',
    'Brunei Darussalam': 'Brunei',
    'Democratic Republic of the Congo': 'Congo, Democratic Republic of (Zaire)',
    'Cambodia': 'Cambodia (Kampuchea)',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Former Czechoslovakia': 'Czechoslovakia',
    'Iran, Islamic Republic of': 'Iran (Persia)',
    'Italy': 'Italy/Sardinia',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Burkina Faso': 'Burkina Faso (Upper Volta)',
    'Republic of North Macedonia': 'Macedonia (FYROM/North Macedonia)',
    'Myanmar': 'Myanmar (Burma)',
    'Syrian Arab Republic': 'Syria',
    'Yemen: Former Yemen Arab Republic': 'Yemen (Arab Republic of Yemen)',
    'Belarus': 'Belarus (Byelorussia)',
    'Republic of Moldova': 'Moldova',
    'Romania': 'Rumania',
    'China (mainland)': 'China',
    'Türkiye': 'Turkey (Ottoman Empire)',
    'Suriname': 'Surinam',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Republic of Korea': 'Korea, Republic of',
    'Madagascar': 'Madagascar (Malagasy)',
    'Zimbabwe': 'Zimbabwe (Rhodesia)',
    "Democratic People's Republic of Korea": "Korea, People's Republic of",
    'Bolivia (Plurinational State of)': 'Bolivia',
    'United Republic of Tanzania: Zanzibar': 'Zanzibar',
    'Former Yugoslavia': 'Yugoslavia',
    'United Republic of Tanzania: Mainland': 'Tanzania (Tanganyika)',
    'Czechia': 'Czech Republic',
    'United States': 'United States of America',
    'Sri Lanka': 'Sri Lanka (Ceylon)',
    'Kingdom of Eswatini': 'Swaziland (Eswatini)',
    }

ag_df['Country/Area'] = ag_df['Country/Area'].replace(mapping_dict)
ag_df = ag_df.rename(columns={'Country/Area': 'country', 'Year': 'year', 'Agriculture, hunting, forestry, fishing (ISIC A-B)': 'Value'})
ag_df = ag_df[['country', 'year', 'Value']]

print(ag_df)


panel = initalize_state_onset_panel(panel_start_year=ag_df['year'].min(),
                                    panel_end_year=2023,
                                    telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_nino34_res0.2_19502023.nc',
                                    pop_path = '/Users/tylerbagwell/Desktop/cccv_data/gpw-v4-population-count-rev11_totpop_15_min_nc/gpw_v4_population_count_rev11_15_min.nc',
                                    clim_index='nino34',
                                    plot_telecon=True)

panel = panel.merge(ag_df, on=['country', 'year'], how='left')


panel.dropna(subset=['Value'], inplace=True)
panel = panel.reset_index(drop=True)
panel = panel.drop('geometry', axis=1)
print(panel)

panel = panel[panel['country'] != 'Barbados']
panel = panel[panel['country'] != 'Comoros']
panel = panel[panel['country'] != 'Estonia']
panel = panel[panel['country'] != 'Ethiopia']
panel = panel[panel['country'] != 'Latvia']
panel = panel[panel['country'] != 'Lithuania']
panel = panel[panel['country'] != 'Maldives']
panel = panel[panel['country'] != 'Malta']
panel = panel[panel['country'] != 'Montenegro']
panel = panel[panel['country'] != 'Singapore']
panel = panel[panel['country'] != 'Sudan']
panel = panel[panel['country'] != 'Trinidad and Tobago']
panel = panel[panel['country'] != 'Yemen (Arab Republic of Yemen)']
panel = panel[panel['country'] != 'Yugoslavia']
panel = panel[panel['country'] != 'Serbia']
panel = panel[panel['country'] != 'Czechoslovakia']

panel.to_csv('/Users/tylerbagwell/Desktop/panel_datasets/ag_panels/UNNA_agrivalueadded_nino34.csv', index=False)