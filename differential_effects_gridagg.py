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

# dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv')
# print(dat)

# dat = dat[dat['psi'] > 0.4]

# stddev = np.std(dat['cindex_lag0y'])
# print(stddev)

# #1
# mask1 = dat['psi_tp_directional'] > +0.0
# anom1 = dat.loc[mask1]
# mask1 = (
#     (anom1['cindex_lag0y'] > +1.0 * stddev)
# )
# anom1 = anom1.loc[mask1]

# anom1_agg = anom1.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'mean',
# }).reset_index()

# #2
# mask2 = dat['psi_tp_directional'] < -0.0
# anom2 = dat.loc[mask2]
# mask2 = (
#     (anom2['cindex_lag0y'] < -1.0 * stddev)
# )
# anom2 = anom2.loc[mask2]

# anom2_agg = anom2.groupby('loc_id').agg({
#     'psi': 'first',
#     'psi_tp_directional': 'first',
#     'tp_anom':'mean',
# }).reset_index()

# mean1 = np.mean(anom1_agg['tp_anom'])
# mean2 = np.mean(anom2_agg['tp_anom'])

# print(anom1_agg.shape[0])
# print(anom2_agg.shape[0])
# rr = anom2_agg.shape[0]/anom1_agg.shape[0]


# x1 = anom1_agg["tp_anom"]
# x2 = anom2_agg["tp_anom"]

# # 1) compute a shared bins array using Scott’s rule on the combined data
# combined = np.concatenate([x1, x2])
# bin_edges = np.histogram_bin_edges(combined, bins="scott")

# # 2) plot both with the same bin_edges
# plt.figure(figsize=(5,4))
# plt.hist(x1,
#          bins=bin_edges,
#          alpha=0.4,
#          label="Pos. corr. w/ NINO3",
#          color="darkorange",
#          density=True,
#          edgecolor="darkorange")
# plt.hist(x2,
#          bins=bin_edges,
#          alpha=0.4,
#          label="Neg. corr. w/ NINO3",
#          color="blue",
#          density=True,
#          edgecolor="blue")
# plt.axvline(mean1, color='darkorange', linestyle='--', linewidth=1.5, label=f'{mean1:.2f}')
# plt.axvline(mean2, color='blue', linestyle=':', linewidth=1.5, label=f'{mean2:.2f}')
# plt.legend(title='Teleconnected regions whose\nprecipitation anomalies are:')
# plt.xlabel('Precipitation anomaly in s.d.')
# plt.ylabel('Density')
# plt.title('DMI induced DRYING')
# plt.xlim(-1.75, +1.75)

# plt.tight_layout()
# plt.show()
# sys.exit()


# ### --- PLOT 1
# plt.figure()
# plt.hist(anom1_agg['tp_anom'], bins='scott', alpha=0.4, label='Pos. corr. w/ NINO3', color='darkorange', density=True, edgecolor='darkorange')
# plt.hist(anom2_agg['tp_anom'], bins='scott', alpha=0.4, label='Neg. corr. w/ NINO3', color='blue', density=True, edgecolor='blue')
# plt.axvline(mean1, color='darkorange', linestyle='--', linewidth=1.5, label=f'{mean1:.2f}')
# plt.axvline(mean2, color='blue', linestyle=':', linewidth=1.5, label=f'{mean2:.2f}')
# # plt.xlim(-4.0,+4.0)
# plt.legend(title='Teleconnected regions whose\nprecipitation anomalies are:')
# plt.xlabel('dryer  ←  Avg. Precipitation anomaly (s.d.)  →  wetter')
# plt.ylabel('Density')
# plt.title('DMI induced DRYING')

# plt.tight_layout()
# # plt.savefig('/Users/tylerbagwell/Desktop/DMI_drying.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()


# sys.exit()
# ### --- PLOT 2
# conflict_dat = dat[dat['conflict_count']>=1]
# # conflict_dat = conflict_dat[conflict_dat['psi_tp_directional'] < +0.0]
# print(conflict_dat)

# conflict_teleconnected = conflict_dat[conflict_dat['psi']>+0.5]

# plt.figure()
# plt.hist(conflict_dat['tp_anom'], bins='scott', alpha=0.5, color='darkorange', density=False)
# plt.hist(conflict_teleconnected['tp_anom'], bins='scott', alpha=0.5, color='purple', edgecolor='black', density=False)
# plt.legend()
# plt.xlabel('anom_tp')
# plt.show()

# print(np.mean(conflict_dat['tp_anom']))
# print(np.mean(conflict_teleconnected['tp_anom']))

# sys.exit()



#####################
#####################


from scipy.stats import linregress
from statsmodels.nonparametric.smoothers_lowess import lowess

dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv')
# print(dat)

stddev = np.std(dat['cindex_lag0y'])
# print(stddev)

dat = dat[dat['psi'] > 0.4]

mask_pos = dat['cindex_lag0y'] > +1.0 * stddev
dat_pos = dat.loc[mask_pos]
agg_pos = dat_pos.groupby('loc_id').agg({
    'psi': 'first',
    'psi_tp_directional': 'first',
    'tp_anom':'median',
}).reset_index()
agg_pos.rename(columns={'tp_anom': 'tp_anom_pos'}, inplace=True)

mask_neg = dat['cindex_lag0y'] < -1.0 * stddev
dat_neg = dat.loc[mask_neg]
agg_neg = dat_neg.groupby('loc_id').agg({
    'psi': 'first',
    'psi_tp_directional': 'first',
    'tp_anom':'median',
}).reset_index()
agg_neg.rename(columns={'tp_anom': 'tp_anom_neg'}, inplace=True)

dat_posneg = agg_pos.merge(agg_neg, on='loc_id', how='inner')
dat_posneg['tp_anom_diff'] = np.abs(dat_posneg['tp_anom_pos'] - dat_posneg['tp_anom_neg'])
dat_posneg.drop(columns=['psi_y','psi_tp_directional_y'], inplace=True)
dat_posneg.rename(columns={'psi_x': 'psi', 'psi_tp_directional_x':'psi_tp_directional'}, inplace=True)


x = dat_posneg['psi']
y = dat_posneg['tp_anom_diff']

corr = np.corrcoef(x, y)[0, 1]
print(corr)

res = linregress(x, y)
x_line = np.array([x.min(), x.max()])
y_line = res.slope * x_line + res.intercept

plt.figure(figsize=(4, 3.5))  

plt.scatter(x, y, color='snow', alpha=0.5, edgecolor='k', s=20)
plt.plot(x_line, y_line, color='blue', linestyle='-', linewidth=1.5)

loess_smoothed = lowess(endog=y, exog=x, frac=0.3, return_sorted=True)
x_loess, y_loess = loess_smoothed[:, 0], loess_smoothed[:, 1]
plt.plot(x_loess, y_loess, color="r", lw=2, linestyle='--')

plt.text(
    0.05, 0.95,                     # x, y in axes fraction coords
    f"corr. = {corr:.2f}",             # the text
    transform=plt.gca().transAxes, # use axes coords
    verticalalignment="top"        # so text starts at y=0.95 and goes down
)

plt.xlabel(r"Teleconnection strength ($\Psi$)")
plt.ylabel(r"| Median($P_{anom}^{+IOD}$) - Median($P_{anom}^{-IOD}$) | (s.d.)")
plt.title("IOD, Global grid cells")
plt.tight_layout()
plt.show()


######

pos = dat_posneg.loc[dat_posneg["psi_tp_directional"] >= 0, "tp_anom_diff"]
neg = dat_posneg.loc[dat_posneg["psi_tp_directional"] <  0, "tp_anom_diff"]

# compute shared bin edges using Scott’s rule on the combined data
combined = np.concatenate([pos, neg])
bin_edges = np.histogram_bin_edges(combined, bins="scott")

# plot
plt.figure(figsize=(6,5))
plt.hist(
    pos,
    bins=bin_edges,
    color="red",
    alpha=0.5,
    label="ψ ≥ 0",
    density=True,
    edgecolor="k"
)
plt.hist(
    neg,
    bins=bin_edges,
    color="blue",
    alpha=0.5,
    label="ψ < 0",
    density=True,
    edgecolor="k"
)

plt.xlabel("tp_anom_diff")
plt.ylabel("Density")
plt.title("tp_anom_diff Distribution by ψ sign")
plt.legend(title="psi_tp_directional")
plt.tight_layout()
plt.show()



#####################
#####################

# from statsmodels.nonparametric.smoothers_lowess import lowess

# cmap = plt.get_cmap('PuOr')
# num_colors = 9
# levels = np.linspace(0, 1, num_colors)
# colors = [cmap(level) for level in levels]

# dat = pd.read_csv('/Users/tylerbagwell/Desktop/YearlyAnom_tp_DMItype2_Global_square4_19502023.csv')
# print(dat)

# dat = dat[dat['psi'] > 0.4]

# stddev = np.std(dat['cindex_lag0y'])
# print(stddev)

# df = dat.copy()

# # make masks
# mask_pos = df["psi_tp_directional"] >= 0
# mask_neg = df["psi_tp_directional"] <  0

# x_pos = df.loc[mask_pos, "cindex_lag0y"]
# y_pos = df.loc[mask_pos, "tp_anom"]

# x_neg = df.loc[mask_neg, "cindex_lag0y"]
# y_neg = df.loc[mask_neg, "tp_anom"]

# plt.figure(figsize=(4,3.5))

# # scatter
# plt.scatter(x_pos, y_pos, color=colors[2],  alpha=0.15, s=10, label="ψ ≥ 0")
# plt.scatter(x_neg, y_neg, color=colors[6],  alpha=0.15, s=10, label="ψ <  0")


# # LOWESS for ψ ≥ 0
# lo_pos = lowess(endog=y_pos, exog=x_pos, frac=0.5, return_sorted=True)
# plt.plot(lo_pos[:,0], lo_pos[:,1], color=colors[1],   lw=2, label="LOESS ψ ≥ 0")

# # LOWESS for ψ < 0
# lo_neg = lowess(endog=y_neg, exog=x_neg, frac=0.5, return_sorted=True)
# plt.plot(lo_neg[:,0], lo_neg[:,1], color=colors[7],  lw=2, label="LOESS ψ <  0")

# plt.axhline(0, color='gray', linestyle='--', linewidth=1.5, zorder=0)

# plt.ylim(-4.0, +6.0)

# # annotate & finish

# plt.xlabel("cindex_lag0y")
# plt.ylabel("tp_anom")
# # plt.title("tp_anom vs cindex_lag0y with LOESS by ψ sign")
# plt.legend()
# plt.tight_layout()
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