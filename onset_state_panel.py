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
from shapely.geometry import mapping
import statsmodels.api as sm
from prepare_index import *

print('\n\nSTART ---------------------\n')


# read in country border data
shape_path = "data/CShapes-2.0/CShapes-2.0.shp"
gdf = gpd.read_file(shape_path)

# need to compute the weighted population grid for each country
# import rasterio

path = '/Users/tylerbagwell/Downloads/gpw-v4-population-count-rev11_totpop_30_min_nc/gpw_v4_population_count_rev11_30_min.nc'
# pop = xr.open_dataarray(path)

ds = xr.open_dataset(path)


# Print a summary of the dataset to inspect dimensions and variables
# print(ds)

# # Example: List all data variables and coordinates
# print("Data Variables:", list(ds.data_vars))
# print("Coordinates:", list(ds.coords))

print(ds['raster'])