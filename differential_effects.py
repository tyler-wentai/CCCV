import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point
import shapely

print('\n\nSTART ---------------------\n')


# dat = pd.read_csv('/Users/tylerbagwell/Desktop/differential_MEANanom_tp.csv')
# print(dat)

# mask1 = dat['cindex_lag0y'] >= +0.75
# mask2 = dat['cindex_lag0y'] <= -0.75
# anom_pos = dat.loc[mask1, 'anom_tp']
# anom_neg = dat.loc[mask2, 'anom_tp']

# # Plot overlapping histograms
# plt.figure()
# plt.hist(anom_pos, bins=30, alpha=0.5, label='cindex_lag0y ≥ 0', color='darkorange', density=True)
# plt.hist(anom_neg, bins=30, alpha=0.5, label='cindex_lag0y < 0', color='blue', density=True)
# plt.legend()
# plt.xlabel('anom_tp')
# plt.ylabel('Frequency')
# plt.title('Histogram of anom_tp by cindex_lag0y Sign')
# plt.show()

# sys.exit()

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

psi_threshold = 0.471
gdf = gdf[gdf['psi'] >= psi_threshold]

cols = ['loc_id', 'year', 'conflict_count', 'SOVEREIGNT', 'cindex_lag0y', 'psi', 'geometry']
gdf = gdf[cols]
gdf = gdf.reset_index()

### compute spatially averaged differential teleconnection strength
path_psi = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_tpDMI.nc'

psi = xr.open_dataarray(path_psi)
df_psi = psi.to_dataframe(name='psi').reset_index()
df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['lon'], row['lat']), axis=1)
psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
psi_gdf = psi_gdf[['lat', 'lon', 'psi', 'geometry']]

# check crs
if psi_gdf.crs != gdf.crs:
    psi_gdf = psi_gdf.to_crs(gdf.crs)
    print("Reprojected gdf to match final_gdf CRS.")

joined_gdf = gpd.sjoin(psi_gdf, gdf, how='left', predicate='within')

cleaned_gdf = joined_gdf.dropna(subset=['loc_id'])
cleaned_gdf = cleaned_gdf.reset_index(drop=True)


grouped = joined_gdf.groupby('loc_id')
mean_psi = grouped['psi_left'].mean().reset_index() # Computing aggregated psi using the MAX of all psis in polygon

gdf = gdf.merge(mean_psi, on='loc_id', how='left')
gdf.rename(columns={'psi_left': 'psi_tp_directional'}, inplace=True)

# df = gdf.drop(columns="geometry")
# df.to_csv("/Users/tylerbagwell/Desktop/Psi_tp_directional_Onset_Count_Global_DMItype2_square4_wGeometry.csv", index=False)
# print(gdf)


###
anom_vals = []
for i in gdf.index:
    print("... ", i, " / ", len(gdf.index))
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
df.to_csv("/Users/tylerbagwell/Desktop/differential_MEANanom_tp_threshold_0d471.csv", index=False)