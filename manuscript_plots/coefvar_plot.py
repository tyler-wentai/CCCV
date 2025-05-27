import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point
import shapely
from scipy.signal import detrend
import seaborn as sns
import matplotlib.ticker as mticker

print('\n\nSTART ---------------------\n')

# 1
path_state1 = '/Users/tylerbagwell/Desktop/cccv_data/supplementary_data/PsiCV_Onset_Binary_GlobalState_NINO3type2.csv'
path_grid1 = '/Users/tylerbagwell/Desktop/cccv_data/supplementary_data/PsiCV_Onset_Count_Global_NINO3type2_square4.csv'

dat_state1   = pd.read_csv(path_state1)
dat_grid1    = pd.read_csv(path_grid1)

min_val1 = min(dat_state1['std_psi'].min(), dat_grid1['std_psi'].min())
max_val1 = max(dat_state1['std_psi'].max(), dat_grid1['std_psi'].max())
bins1 = np.linspace(min_val1, max_val1, 11)

med_state1 = dat_state1['std_psi'].median()
med_grid1  = dat_grid1['std_psi'].median()

# 2
path_state2 = '/Users/tylerbagwell/Desktop/cccv_data/supplementary_data/PsiCV_Onset_Binary_GlobalState_DMItype2.csv'
path_grid2 = '/Users/tylerbagwell/Desktop/cccv_data/supplementary_data/PsiCV_Onset_Count_Global_DMItype2_square4.csv'

dat_state2   = pd.read_csv(path_state2)
dat_grid2    = pd.read_csv(path_grid2)

min_val2 = min(dat_state2['std_psi'].min(), dat_grid2['std_psi'].min())
max_val2 = max(dat_state2['std_psi'].max(), dat_grid2['std_psi'].max())
bins2 = np.linspace(min_val2, max_val2, 11)

med_state2 = dat_state2['std_psi'].median()
med_grid2  = dat_grid2['std_psi'].median()

# plotting
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

# NINO3
ax = axes[0]
ax.hist(dat_state1['std_psi'], bins=bins1, alpha=0.5,
        edgecolor='white', density=True, label='Aggregated by state')
ax.hist(dat_grid1['std_psi'],  bins=bins1, alpha=0.5,
        edgecolor='white', density=True, label='Aggregated by grid cell (4°×4°)')
ax.axvline(med_state1, color='C0', linestyle='--', linewidth=2,
           label=f'State median = {med_state1:.3f}')
ax.axvline(med_grid1,  color='C1', linestyle='--', linewidth=2,
           label=f'Grid cell median = {med_grid1:.3f}')
ax.set_xlabel('Standard deviation of within-unit teleconnection strengths')
ax.set_ylabel('Density')
ax.set_title('NINO3\nStandard deviation of within-unit teleconnection strengths', fontsize=10)
ax.text(0.05, 0.9, 'A', transform=ax.transAxes, fontsize=18)
ax.legend(fontsize=8)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_xlim(-0.02,+0.35)

# DMI
ax = axes[1]
ax.hist(dat_state2['std_psi'], bins=bins2, alpha=0.5,
        edgecolor='white', density=True, label='Aggregated by state')
ax.hist(dat_grid2['std_psi'],  bins=bins2, alpha=0.5,
        edgecolor='white', density=True, label='Aggregated by grid cell (4°×4°)')
ax.axvline(med_state2, color='C0', linestyle='--', linewidth=2,
           label=f'State median = {med_state2:.3f}')
ax.axvline(med_grid2,  color='C1', linestyle='--', linewidth=2,
           label=f'Grid cell median = {med_grid2:.3f}')
ax.set_xlabel('Standard deviation of within-unit teleconnection strengths')
ax.set_title('DMI\nStandard deviation of within-unit teleconnection strengths', fontsize=10)
ax.text(0.05, 0.9, 'B', transform=ax.transAxes, fontsize=18)
ax.legend(fontsize=8)
ax.grid(True, axis='y', linestyle='--', alpha=0.7)
ax.set_xlim(-0.02,+0.35)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/cccv_manuscript_plots/supp_hist_std_withinunit_psi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
