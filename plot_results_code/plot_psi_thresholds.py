import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import detrend
from scipy.stats import pearsonr
import pandas as pd
import sys
from datetime import datetime, timedelta
import xarray as xr
import seaborn as sns
import matplotlib.image as mpimg
import geopandas as gpd
from matplotlib.colors import ListedColormap
import regionmask

print('\n\nSTART ---------------------\n')

land_regs   = regionmask.defined_regions.natural_earth_v5_0_0.land_110

#### --- NINO3
ds1 = xr.open_dataset('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiMonthlyspi6_NINO3_FINAL.nc')
var1_name = list(ds1.data_vars)[0]
da1 = ds1[var1_name]

mask1        = land_regs.mask(da1)      # this creates an integer mask: land cells get region IDs ≥0, ocean cells get −1
da1_land     = da1.where(mask1>=0)      # keep only land

vals1 = da1_land.values.ravel()
vals1 = vals1[~np.isnan(vals1)]

mask1   = (~np.isnan(vals1)) & (vals1 != 0)
clean1  = vals1[mask1]
med1    = np.median(clean1)
mean1   = np.mean(clean1)

#### --- DMI
ds2 = xr.open_dataset('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiMonthly_DMI_FINAL.nc')
var2_name = list(ds2.data_vars)[0]
da2 = ds2[var2_name]

mask2        = land_regs.mask(da2)      # this creates an integer mask: land cells get region IDs ≥0, ocean cells get −1
da2_land     = da2.where(mask2>=0)      # keep only land

vals2 = da2_land.values.ravel()
vals2 = vals2[~np.isnan(vals2)]

mask2   = (~np.isnan(vals2)) & (vals2 != 0)
clean2  = vals2[mask2]
med2    = np.median(clean2)
mean2   = np.mean(clean2)

#### --- ANI
ds3 = xr.open_dataset('/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psiMonthly_ANI_FINAL.nc')
var3_name = list(ds3.data_vars)[0]
da3 = ds3[var3_name]

mask3        = land_regs.mask(da3)      # this creates an integer mask: land cells get region IDs ≥0, ocean cells get −1
da3_land     = da3.where(mask3>=0)      # keep only land

vals3 = da3_land.values.ravel()
vals3 = vals3[~np.isnan(vals3)]

mask3   = (~np.isnan(vals3)) & (vals3 != 0)
clean3  = vals3[mask3]
med3    = np.median(clean3)
mean3   = np.mean(clean3)




#  histograms
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 6), sharex=False)

datasets = [
    (clean1, med1, mean1, da1.name, "ENSO (via NINO3, tropical year spans 12 months)"),
    (clean2, med2, mean2, da2.name, "IOD (via DMI, tropical year spans 8 months)"),
    (clean3, med3, mean3, da3.name, "AN (via ANI, tropical year spans 8 months)"),
]

plt.suptitle('Monthly teleconnection strengths for land-based grid points', fontsize=12)

for i, (ax, (data, med, mean, name, title)) in enumerate(zip(axs, datasets)):
    sns.histplot(
        data,
        stat='density',
        bins=50,
        kde=True,
        line_kws={'color': 'green', 'linewidth': 1},
        ax=ax,
        edgecolor='k'
    )
    ax.axvline(med,  color='r', linestyle=':', linewidth=1.5, label=f'Median = {med:.2f}')
    ax.axvline(mean, color='b', linestyle='-.', linewidth=1.5, label=f'Mean = {mean:.2f}')
    ax.axvline(0.45, color='g', linestyle='--', linewidth=1.5, label=f'Monthly threshold')
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(title, fontsize=10, loc='left')
    # ax.set_xlim([0.2, 1.1])
    ax.legend(frameon=False, fontsize=9)

    # only give the xlabel to the bottom axis
    if i == len(axs) - 1:
        ax.set_xlabel(f'Monthly teleconnection strength', fontsize=10)
    else:
        ax.set_xlabel('')             # remove the label
        # ax.tick_params(labelbottom=False)  # also hide its tick labels


plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/Hist_MonthlyPsis_threshold.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
