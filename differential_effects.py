import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point

print('\n\nSTART ---------------------\n')


dat = pd.read_csv('/Users/tylerbagwell/Desktop/differential_MEANanom_tp.csv')
print(dat)

mask = dat['cindex_lag0y'] >= 0
anom_pos = dat.loc[mask, 'anom_tp']
anom_neg = dat.loc[~mask, 'anom_tp']

# Plot overlapping histograms
plt.figure()
plt.hist(anom_pos, bins=30, alpha=0.5, label='cindex_lag0y ≥ 0', color='darkorange', density=True)
plt.hist(anom_neg, bins=30, alpha=0.5, label='cindex_lag0y < 0', color='blue', density=True)
plt.legend()
plt.xlabel('anom_tp')
plt.ylabel('Frequency')
plt.title('Histogram of anom_tp by cindex_lag0y Sign')
plt.show()

sys.exit()

path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/tp_anom_ERA5_0d5_19502023_wTimeLatLon.nc'

var1 = xr.open_dataset(path)
# var1_yr = var1.resample(time="1Y").mean()
# var1_yr = var1_yr.groupby("time.year").mean(dim="time")

var1_yr = var1.sel(time=var1.time.dt.month.isin(range(5,13))) #grab months may-dec
var1_yr = var1_yr.groupby("time.year").mean(dim="time")


###
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4_wGeometry.csv'
df = pd.read_csv(path)
df['geometry'] = df['geometry'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)

psi_threshold = 0.50
gdf = gdf[gdf['psi'] >= psi_threshold]

cols = ['loc_id', 'year', 'conflict_count', 'SOVEREIGNT', 'cindex_lag0y', 'psi', 'geometry']
gdf = gdf[cols]
gdf = gdf.reset_index()




###

anom_vals = []
for i in gdf.index:
    print("... ", i)
    yr = gdf['year'].iloc[i]
    anom = var1_yr.sel(year=yr)

    varname = list(anom.data_vars)[0]
    anom = anom[varname]

    anom_df = anom.to_dataframe(name="value").reset_index()
    anom_df["geometry"] = [Point(xy) for xy in zip(anom_df.lon, anom_df.lat)]
    anom_gdf = gpd.GeoDataFrame(anom_df, geometry="geometry")
    anom_gdf.set_crs("EPSG:4326", inplace=True)

    anom_gdf = anom_gdf[["year", "value", "geometry"]]
    joined_gdf = gpd.sjoin(anom_gdf, gdf.iloc[[i]], how='left', predicate='within')
    cleaned_gdf = joined_gdf.dropna(subset=['loc_id'])

    mean_anom = cleaned_gdf['value'].mean()
    print("...... ", np.round(mean_anom,3))
    anom_vals.append(mean_anom)

gdf['anom_tp'] = anom_vals

df = gdf.drop(columns="geometry")
df.to_csv("/Users/tylerbagwell/Desktop/differential_MEANanom_tp.csv", index=False)