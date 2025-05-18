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

print('\n\nSTART ---------------------\n')

############ COMPUTE ############

path_var1_yr = '/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/ERA5_t2m_YearlyMeanMayDec_0d50_19502023.nc'
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4_wGeometry.csv'
path_psi = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_t2mDMI.nc'
varname = 't2m'


### --- 1: This var1_yr is dervied from compute_yr_anom.py
var1_yr = xr.open_dataset(path_var1_yr)


###
df = pd.read_csv(path)
df['geometry'] = df['geometry'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)

cols = ['loc_id', 'year', 'conflict_count', 'SOVEREIGNT', 'cindex_lag0y', 'psi', 'geometry']
gdf = gdf[cols]
gdf = gdf.reset_index()

### compute spatially averaged differential teleconnection strength
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
mean_psi = grouped['psi_left'].mean().reset_index() # Computing aggregated psi using the MEAN of all psis in polygon

gdf = gdf.merge(mean_psi, on='loc_id', how='left')
gdf.rename(columns={'psi_left': str('psi_'+varname+'_directional')}, inplace=True)


gdf_agg = gdf.groupby('loc_id', as_index=False).first()
gdf_agg = gdf_agg[['loc_id', 'geometry']]
gdf_agg = gdf_agg.set_crs("EPSG:4326", allow_override=True)


###
min_year = np.min(list(set(gdf['year'])))
max_year = np.max(list(set(gdf['year'])))

print(min_year, max_year)


###
anom_by_year = {}
for yr in set(gdf['year']):
    print("...", yr)
    anom = var1_yr.sel(year=yr)

    varname = list(anom.data_vars)[0]
    anom = anom[varname]

    anom_df = anom.to_dataframe(name="value").reset_index()
    anom_df["geometry"] = [Point(xy) for xy in zip(anom_df.longitude, anom_df.latitude)]
    anom_gdf = gpd.GeoDataFrame(anom_df, geometry="geometry")
    anom_gdf.set_crs("EPSG:4326", inplace=True)

    anom_gdf = anom_gdf[["year", "value", "geometry"]]
    anom_by_year[yr] = anom_gdf


###
all_dfs = []
for i in gdf_agg.index:
    if (i % 1 == 0):
        print("... ", i, " / ", len(gdf_agg.index))

    yy = []
    for yr in range(min_year, max_year+1):
        var_gdf         = anom_by_year[yr]
        joined_gdf      = gpd.sjoin(var_gdf, gdf_agg.iloc[[i]], how='left', predicate='within')
        cleaned_gdf     = joined_gdf.dropna(subset=['loc_id'])
        spat_avg_val    = cleaned_gdf['value'].mean()
        yy.append(spat_avg_val)

    years = np.arange(min_year, max_year+1)
    yy_dt = detrend(yy)
    yy_dt = np.array(yy_dt)
    mu = yy_dt.mean()
    sigma = yy_dt.std(ddof=1)   # population stdev; use ddof=1 for sample stdev
    yy_std = (yy_dt - mu) / sigma

    df_anom = pd.DataFrame({
    str(varname):           yy,
    str(varname+'_anom'):   yy_std,
    'year':                 years,
    'loc_id':               gdf_agg.iloc[i]['loc_id']      # this string is broadcast to every row
    })

    all_dfs.append(df_anom)


df_all = pd.concat(all_dfs, ignore_index=True)
print(df_all)

merged = gdf.merge(
    df_all,
    on=['loc_id','year'],
    how='left'        # or 'left', 'right', 'outer' depending on what you need
)

merged = merged.drop(columns="geometry")

df = merged.sort_values(['loc_id', 'year'])
df[str('d'+varname)] = df.groupby('loc_id')[varname].diff()
df[str('d'+varname+'_anom')] = (
    df
    .groupby('loc_id')[str('d'+varname)]
    .transform(lambda x: (x - x.mean()) / x.std())
)

path_str = "/Users/tylerbagwell/Desktop/SpatialAgg_MayDecAnnual_" + varname +"_DMItype2_Global_square4_19502023.csv"
df.to_csv(path_str, index=False)