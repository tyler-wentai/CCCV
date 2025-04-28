import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from build_panel_state import initalize_state_onset_panel

print('\n\nSTART ---------------------\n')

ag_path = '/Users/tylerbagwell/Desktop/cccv_data/ag_data/maize_FAOSTAT_data_en_2-23-2025.csv'
ag_df = pd.read_csv(ag_path)

mapping_dict = {
    'OldName1': 'NewName1',
    'Syrian Arab Republic': 'Syria',
    'China, mainland': 'China',
    'Bolivia (Plurinational State of)': 'Bolivia',
    'Yemen': "Yemen, People's Republic of",
    'United Republic of Tanzania': 'Tanzania (Tanganyika)',
    'Bosnia and Herzegovina': 'Bosnia-Herzegovina',
    'Eswatini': 'Swaziland (Eswatini)',
    'Madagascar': 'Madagascar (Malagasy)',
    'Kyrgyzstan': 'Kyrgyz Republic',
    'Venezuela (Bolivarian Republic of)': 'Venezuela',
    'Myanmar': 'Myanmar (Burma)',
    'Türkiye': 'Turkey (Ottoman Empire)',
    'United Kingdom of Great Britain and Northern Ireland': 'United Kingdom',
    'Romania': 'Rumania',
    'China, Taiwan Province of': 'Taiwan',
    'Belarus': 'Belarus (Byelorussia)',
    "Democratic People's Republic of Korea": "Korea, People's Republic of",
    'Netherlands (Kingdom of the)': 'Netherlands',
    'North Macedonia': 'Macedonia (FYROM/North Macedonia)',
    'Democratic Republic of the Congo': 'Congo, Democratic Republic of (Zaire)',
    'Italy': 'Italy/Sardinia',
    'Republic of Moldova': 'Moldova',
    'Serbia and Montenegro': 'Montenegro',
    'Republic of Korea': 'Korea, Republic of',
    'Czechia': 'Czech Republic',
    'Zimbabwe': 'Zimbabwe (Rhodesia)',
    'Iran (Islamic Republic of)': 'Iran (Persia)'
    }

ag_df['Area'] = ag_df['Area'].replace(mapping_dict)
ag_df = ag_df.rename(columns={'Area': 'country', 'Year': 'year'})
ag_df = ag_df[['country', 'year', 'Value']]

print(ag_df)

panel = initalize_state_onset_panel(panel_start_year=ag_df['year'].min(),
                                    panel_end_year=2023,
                                    telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_ANI_cai_0d5.nc',
                                    pop_path = '/Users/tylerbagwell/Desktop/cccv_data/gpw-v4-population-count-rev11_totpop_15_min_nc/gpw_v4_population_count_rev11_15_min.nc',
                                    clim_index='ani',
                                    plot_telecon=True)

panel = panel.merge(ag_df, on=['country', 'year'], how='left')


panel.dropna(subset=['Value'], inplace=True)
panel = panel.reset_index(drop=True)
panel = panel.drop('geometry', axis=1)
print(panel)

panel = panel[panel['country'] != 'Malta']
panel = panel[panel['country'] != 'Somalia']

panel.to_csv('/Users/tylerbagwell/Desktop/panel_datasets/ag_panels/FAOSTAT_maizeyield_ani.csv', index=False)