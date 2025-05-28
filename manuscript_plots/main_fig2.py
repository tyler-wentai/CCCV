import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
from calc_annual_index import *


print('\n\nSTART ---------------------\n')

# onsets
onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
df_onset = pd.read_csv(onset_path)    
gdf_onset = gpd.GeoDataFrame(
    df_onset, 
    geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
    crs="EPSG:4326"
)

gdf_onset['year'] = pd.to_datetime(gdf_onset['year'], format='%Y')
yearly_counts = (
    gdf_onset
      .groupby(gdf_onset['year'].dt.year)
      .size()
      .reset_index(name='n_events')
)
print(yearly_counts['n_events'].sum(), 'total conflict onsets')


start_year, end_year = 1950, 2023

nino3_monthly = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
dmi_monthly = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))

nino3_yearly    = compute_annualized_index('nino3', start_year, end_year+1)
dmi_yearly      = compute_annualized_index('dmi', start_year, end_year)

nino3_std   = nino3_yearly['cindex'].std()
dmi_std     = dmi_yearly['cindex'].std()

sigma_nino3 = nino3_monthly['ANOM'].std()
sigma_dmi = dmi_monthly['ANOM'].std()

yearly_counts['year'] = pd.to_datetime(yearly_counts['year'].astype(str), format='%Y')
nino3_yearly['year'] = pd.to_datetime(nino3_yearly['year'].astype(str), format='%Y')
dmi_yearly['year'] = pd.to_datetime(dmi_yearly['year'].astype(str), format='%Y')

import seaborn as sns
cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 3
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

# assign colors
def pick_color_nino3(x):
    if x < -sigma_nino3:
        return 'blue'
    elif x > sigma_nino3:
        return 'red'
    else:
        return 'gray'
def pick_color_dmi(x):
    if x < -sigma_dmi:
        return colors[0]
    elif x > sigma_dmi:
        return colors[2]
    else:
        return 'gray'
colors_nino3 = nino3_yearly['cindex'].map(pick_color_nino3)
colors_dmi = dmi_yearly['cindex'].map(pick_color_dmi)


# build legend patches
legend_patches_nino3 = [
    Patch(color='red',   alpha=0.6, label='El Niño'),
    Patch(color='gray',  alpha=0.6, label='Neutral'),
    Patch(color='blue',  alpha=0.6, label='La Niña'),
]
legend_patches_dmi = [
    Patch(color=colors[2],   alpha=0.6, label='+IOD'),
    Patch(color='gray',  alpha=0.6, label='Neutral'),
    Patch(color=colors[0],  alpha=0.6, label='-IOD'),
]

# create two stacked subplots, sharing the x-axis
fig, (ax1, ax2, ax3) = plt.subplots(
    nrows=3, ncols=1,
    figsize=(7.0, 7.5),
    sharex=False
)
fig.subplots_adjust(hspace=0.4)

# — Panel A —
ax1.bar(
    yearly_counts['year'],
    yearly_counts['n_events'],
    width=300,
    alpha=0.7,
    color='green',
    zorder=0
)
ax1.axhline(0, color='black', linewidth=1.0, linestyle='-')
ax1.set_ylabel('No. of conflict onsets')
ax1.set_title('State-based Armed Conflict Onsets (1950-2023), N=555')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.text(
    0.95, 0.1, 'A',
    transform=ax1.transAxes,
    fontsize=18,
    bbox=dict(
        boxstyle='square,pad=0.3',
        facecolor='white',
        edgecolor='black',
        linewidth=1
    )
)

# — Panel B —
ax2.plot(
    nino3_yearly['year'],
    nino3_yearly['cindex'],
    color='black',
    linewidth=1.5,
    zorder=1
)
ax2.bar(
    nino3_yearly['year'],
    nino3_yearly['cindex'],
    width=300,
    alpha=0.7,
    color=colors_nino3,
    zorder=0
)
ax2.axhline(0, color='black', linewidth=1.0, linestyle='-')
ax2.set_ylabel('NDJ Averaged NINO3 (°C)')
ax2.set_title('Annualized Index for The El-Nino Southern Oscillation')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.text(
    0.95, 0.1, 'B',
    transform=ax2.transAxes,
    fontsize=18,
    bbox=dict(
        boxstyle='square,pad=0.3',
        facecolor='white',
        edgecolor='black',
        linewidth=1
    )
)
ax2.legend(
    handles=legend_patches_nino3,
    loc='upper left',
    frameon=True,
    fontsize=9
)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# — Panel C —
ax3.plot(
    dmi_yearly['year'],
    dmi_yearly['cindex'],
    color='black',
    linewidth=1.5,
    zorder=1
)
ax3.bar(
    dmi_yearly['year'],
    dmi_yearly['cindex'],
    width=300,
    alpha=0.7,
    color=colors_dmi,
    zorder=0
)
ax3.axhline(0, color='black', linewidth=1.0, linestyle='-')
ax3.set_ylabel('SON Averaged DMI (°C)')
ax3.set_title('Annualized Index for The Indian Ocean Dipole')
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.text(
    0.95, 0.1, 'C',
    transform=ax3.transAxes,
    fontsize=18,
    bbox=dict(
        boxstyle='square,pad=0.3',
        facecolor='white',
        edgecolor='black',
        linewidth=1
    )
)
ax3.set_xlabel('Year')
ax3.legend(
    handles=legend_patches_dmi,
    loc='upper left',
    frameon=True,
    fontsize=9
)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))



plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/manuscript_plots/Main_fig2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()