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

# dat = pd.read_csv('/Users/tylerbagwell/Desktop/Psi_tp_directional_Onset_Count_Global_DMItype2_square4_wGeometry.csv')

# dat_agg =dat.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
# }).reset_index()

# psi_threshold = 0.25
# mask = dat_agg['psi'] > psi_threshold
# dat_agg = dat_agg.loc[mask]

# dat_agg['psi_tp_directional'] = np.abs(dat_agg['psi_tp_directional'])

# plt.figure()
# plt.scatter(dat_agg['psi_tp_directional'], dat_agg['psi'])
# plt.show()

# print(np.corrcoef(dat_agg['psi_tp_directional'], dat_agg['psi']))

# sys.exit()

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv')
print(dat)

dat = dat[dat['psi'] > 0.405640972]

stddev = np.std(dat['cindex_lag0y'])
print(stddev)


mask1 = dat['psi_tp_directional'] > +0.1
anom1 = dat.loc[mask1]
mask1 = anom1['cindex_lag0y'] < -1.0*stddev
anom1 = anom1.loc[mask1, 'tp_anom']

# mask2 = dat['psi_tp_directional'] < 0
# anom2 = dat.loc[mask2]
# mask2 = anom2['cindex_lag0y'] < +1.5*stddev
# anom2 = anom2.loc[mask2, 'anom_tp']

mask2 = dat['psi_tp_directional'] < -0.1
anom2 = dat.loc[mask2]
mask2 = anom2['cindex_lag0y'] > +1.0*stddev
anom2 = anom2.loc[mask2, 'tp_anom']

mean1 = np.mean(anom1)
mean2 = np.mean(anom2)


# Plot overlapping histograms
plt.figure()
# plt.axvline(0.0, color='gray', linestyle='-', linewidth=2)
plt.hist(anom1, bins='scott', alpha=0.5, label='Pos. corr. w/ NINO3', color='darkorange', density=True)
plt.hist(anom2, bins='scott', alpha=0.5, label='Neg. corr. w/ NINO3', color='blue', density=True)
plt.axvline(mean1, color='darkorange', linestyle='--', linewidth=0.5, label=f'{mean1:.2f}')
plt.axvline(mean2, color='blue', linestyle=':', linewidth=0.5, label=f'{mean2:.2f}')
# plt.xlim(-4.0,+4.0)
plt.legend(title='Teleconnected regions whose\nprecipitation anomalies are:')
plt.xlabel('dryer  ←  Precipitation anomaly (s.d.)  →  wetter')
# plt.axvspan(+0.0, +4.0, color='green', alpha=0.10, edgecolor='none', linewidth=0.0, zorder=0)
# plt.axvspan(+0.0, -4.0, color='brown', alpha=0.10, edgecolor='none', linewidth=0.0, zorder=0)
plt.ylabel('Density')
plt.title('NINO3 induced WETTING')

plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/NINO3_wetting.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()

# conflict_dat = dat[dat['conflict_count']==1]
# print(conflict_dat)

# plt.figure()
# plt.hist(conflict_dat['anom_tp'], bins='scott', alpha=0.5, color='darkorange', density=False)
# plt.legend()
# plt.xlabel('anom_tp')
# plt.show()


sys.exit()

############ ------- ############
############ COMPUTE ############
############ ------- ############


### --- 1: This var1_yr is dervied from compute_yr_anom.py
path_var1_yr = '/Users/tylerbagwell/Desktop/cccv_data/processed_climate_data/ERA5_tp_YearlySumMayDec_0d50_19502023.nc'
var1_yr = xr.open_dataset(path_var1_yr)


###
path = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4_wGeometry.csv'
df = pd.read_csv(path)
df['geometry'] = df['geometry'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(df, geometry='geometry')
gdf.set_crs(epsg=4326, inplace=True)

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
    'tp_anom': yy_std,
    'year':    years,
    'loc_id':  gdf_agg.iloc[i]['loc_id']      # this string is broadcast to every row
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
merged.to_csv("/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv", index=False)