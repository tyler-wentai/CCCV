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