import matplotlib.pyplot as plt
import pandas as pd
import sys
from datetime import datetime
import xarray as xr
import geopandas as gpd
import numpy as np

print('\n\nSTART ---------------------\n')

var1_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_LeafAreaIndex_data.nc'
ds1 = xr.open_dataset(var1_path)

var1 = ds1['lai_hv']
var1 = var1.isel(valid_time=-1)

var1.plot()
plt.show()