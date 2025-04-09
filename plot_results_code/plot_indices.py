import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple

import sys
import os

# Determine the parent directory of the current script
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.calc_annual_index import *

print('\n\nSTART ---------------------\n')

start_year = 1950
end_year = 2023

nino3 = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))

dmi  = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))

nino3.rename(columns={'ANOM': 'NINO3'}, inplace=True)
dmi.rename(columns={'ANOM': 'DMI'}, inplace=True)



# Create a figure with 2 subplots (one above the other), sharing the x-axis
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 3.5), sharex=True)

# Plot the NINO3 data on the first subplot
ax1.plot(nino3.index, nino3['NINO3'], linestyle='-', linewidth=2, color='purple', label='NINO3', zorder=1)
ax1.set_ylabel('NINO3')

ax1.axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=2)

# Plot the DMI data on the second subplot
ax2.plot(dmi.index, dmi['DMI'], linestyle='-', linewidth=2, color='green', label='DMI', zorder=1)
ax2.set_xlabel('Year', fontsize=9)
ax2.set_ylabel('DMI', fontsize=9)

# Set the x-axis to show ticks every 10 years
# Since we share the x-axis, set the locator on the bottom axes (ax2)
ax2.xaxis.set_major_locator(mdates.YearLocator(10))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax2.axhline(0, color='black', linewidth=1.5, linestyle='--', zorder=2)


plt.tight_layout()
plt.show()
