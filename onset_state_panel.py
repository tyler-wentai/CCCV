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

# with rasterio.open('data/gl_gpwv3_pcount_90_ascii_quar/glp90ag15.asc') as src:
#     # Read the first band of the dataset as a numpy array
#     array = src.read(1)
#     print(src.meta)

import rasterio
from rasterio.features import shapes
import geopandas as gpd
import numpy as np

# Open the ASCII raster file with rasterio
with rasterio.open('data/gl_gpwv3_pcount_90_ascii_quar/glp90ag15.asc') as src:
    # Read the first band of the raster
    image = src.read(1)
    
    # Create a mask to exclude nodata values (if nodata is defined)
    mask = image != src.nodata if src.nodata is not None else None

    # Extract shapes (polygons) and their associated values from the raster
    results = (
        {'properties': {'value': value}, 'geometry': geometry}
        for geometry, value in shapes(image, mask=mask, transform=src.transform)
    )

    # Convert the generator to a list
    geoms = list(results)

# Create a GeoDataFrame from the list of features, setting the CRS from the source raster
gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

# Display the first few rows of the GeoDataFrame
print(gdf.head())