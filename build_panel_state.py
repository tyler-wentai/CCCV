import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from utils.calc_annual_index import *

print('\n\nSTART ---------------------\n')

def initalize_state_onset_panel(panel_start_year, panel_end_year, telecon_path, pop_path, clim_index, response_var, plot_telecon=False):
    """
    This function initializes the state onset panel. It reads in the country border data, and creates a panel with country-year observations.
    """

    ######## A. INITIALIZE PANEL
    # read in country border data
    shape_path = "data/cshape_files/cshpR.shp"
    df = gpd.read_file(shape_path)

    # Convert the 'start' and 'end' columns to datetime and extract the year
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['year'] = df['start'].dt.year

    # Create a panel with country-year observations
    panel_list = []

    # pre-processing before creation of panel:
    # (1) if multiple border changes occur in a single year, pick the last one in that year
    df = df.sort_values('start', ascending=True).drop_duplicates(subset=['cntry_n', 'year'], keep='last')
    # (2) if the border change occurs after jun 31st in that year, change the border year to the following year
    # NOT IMPLEMENTING (2)

    dataset_max_year = df['end'].dt.year.max() # the last observed year in the cshape dataset

    # Group by country
    for country, group in df.groupby('cntry_n'):

        # Determine the panel year range
        start_year = group['year'].min()
        end_year = group['end'].dt.year.max()

        if (end_year >= dataset_max_year): # case where we extend 2019's (dataset_max_year) geometry to panel_end_year (we assume no major changes)
            end_year = panel_end_year
        elif (end_year < dataset_max_year): # case where the country's existence has ended before dataset_max_year
            end_year = end_year

        # create a DataFrame with one row per year in the panel
        years = pd.DataFrame({'year': range(start_year, end_year + 1)})
        years['cntry_n'] = country

        # sort both DataFrames by year
        group_sorted = group.sort_values('year')
        years_sorted = years.sort_values('year')
        years_sorted['year'] = years_sorted['year'].astype('int64')
        group_sorted['year'] = group_sorted['year'].astype('int64')
        
        # merge using merge_asof to carry forward the last available geometry
        merged = pd.merge_asof(
            years_sorted,
            group_sorted,
            on='year',
            by='cntry_n',
            direction='backward'
        )
        
        panel_list.append(merged)

    # combine the individual country panels
    panel_df = pd.concat(panel_list, ignore_index=True)

    # remove country-years observations before panel_start_year
    panel_df = panel_df[panel_df['year'] >= panel_start_year]
    panel_gdf = gpd.GeoDataFrame(panel_df, geometry='geometry')
    panel_gdf.crs = "EPSG:4326"
    panel_gdf = panel_gdf[['year','cntry_n','gwcode','fid','geometry']]



    ######## D. COMPUTE CONFLICT ONSET COUNT-YEAR
    # load conflict events dataset and convert to GeoDataFrame
    conflictdata_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
    conflict_df = pd.read_csv(conflictdata_path)
    conflict_gdf = gpd.GeoDataFrame(
        conflict_df,
        geometry=gpd.points_from_xy(conflict_df.onset_lon, conflict_df.onset_lat),
        crs='EPSG:4326'
        )

    joined_gdf = gpd.sjoin(conflict_gdf, panel_gdf, how='inner', predicate='within', on_attribute='year')
    joined_gdf = joined_gdf[['conflict_id', 'location', 'year', 'onset_loc', 'onset_loc_prec',
                             'primary_onset', 'geometry', 'cntry_n', 'gwcode', 'fid']]

    conflict_counts = joined_gdf.groupby(['year', 'fid']).size().reset_index(name='conflict_count')

    panel_gdf = pd.merge(panel_gdf, conflict_counts, on=['year', 'fid'], how='left')
    panel_gdf['conflict_count'] = panel_gdf['conflict_count'].fillna(0) # fill missing values with 0, i.e., no conflict

    ######## B. COMPUTE COUNTRY-YEAR TELECONNECTION STRENGTH (POPUlATION WEIGHTED)
    print('...Computing gdf for psi...')
    psi = xr.open_dataarray(telecon_path)
    pop = xr.open_dataarray(pop_path)
    pop2000 = pop.sel(raster=1) #raster 1 corresponds to the year 2000

    psi = psi.rename({'lat': 'latitude', 'lon': 'longitude'})
    lon1 = psi['longitude']
    def convert_longitude(ds):
        longitude = ds['longitude']
        longitude = ((longitude + 180) % 360) - 180
        ds = ds.assign_coords(longitude=longitude)
        return ds
    if lon1.max() > 180:
        psi = convert_longitude(psi)
    psi = psi.sortby('longitude')
    
    # Calculate the spacing for psi's coordinates
    psi_lat_spacing = np.diff(psi.latitude.values).mean()
    psi_lon_spacing = np.diff(psi.longitude.values).mean()

    # Calculate the spacing for pop's coordinates
    pop_lat_spacing = np.diff(pop.latitude.values).mean()
    pop_lon_spacing = np.diff(pop.longitude.values).mean()

    print(f"......psi - Latitude spacing: {psi_lat_spacing}, Longitude spacing: {psi_lon_spacing}")
    print(f"......pop - Latitude spacing: {pop_lat_spacing}, Longitude spacing: {pop_lon_spacing}")

    # Ensure that the DataArray's has 'latitude' and 'longitude' coordinates
    if 'latitude' not in psi.coords or 'longitude' not in psi.coords:
        raise ValueError("psi dataArray must have 'latitude' and 'longitude' coordinates.")
    if 'latitude' not in pop2000.coords or 'longitude' not in pop2000.coords:
        raise ValueError("pop2000 dataArray must have 'latitude' and 'longitude' coordinates.")

    # Interpolate the population dataArray onto the grid of the psi dataArray.
    # pop2000_interp = pop2000

    pop2000_interp = pop2000.interp(
        latitude=psi.latitude,
        longitude=psi.longitude,
        method="linear"
    )

    df_pop2000_interp = pop2000_interp.to_dataframe(name='pop2000').reset_index()
    df_pop2000_interp['geometry'] = df_pop2000_interp.apply(lambda row: shapely.geometry.Point(row['longitude'], row['latitude']), axis=1)
    pop2000_gdf = gpd.GeoDataFrame(df_pop2000_interp, geometry='geometry', crs='EPSG:4326')
    pop2000_gdf = pop2000_gdf[['latitude', 'longitude', 'pop2000', 'geometry']]

    df_psi = psi.to_dataframe(name='psi').reset_index()
    df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['longitude'], row['latitude']), axis=1)
    psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
    psi_gdf = psi_gdf[['latitude', 'longitude', 'psi', 'geometry']]

    # check crs
    if psi_gdf.crs != panel_gdf.crs:
        psi_gdf = psi_gdf.to_crs(panel_gdf.crs)
        print("Reprojected gdf to match final_gdf CRS.")
    if psi_gdf.crs != pop2000_gdf.crs:
        pop2000_gdf = pop2000_gdf.to_crs(psi_gdf.crs)
        print("Reprojected gdf to match final_gdf CRS.")

    unique_fids = panel_gdf[['fid', 'geometry']].drop_duplicates(subset='fid') # the fids are unique to each geometry in df and panel_gdf

    joined_gdf = gpd.sjoin(psi_gdf, unique_fids, how='left', predicate='within')
    joined_gdf = joined_gdf.drop('index_right', axis=1)

    joined_gdf = gpd.sjoin(joined_gdf, pop2000_gdf, how='left', predicate='intersects')
    joined_gdf = joined_gdf.drop('index_right', axis=1)

    joined_gdf['pop_weighted_psi'] = joined_gdf['psi'] * joined_gdf['pop2000']

    joined_gdf = joined_gdf.dropna(subset=['fid'])
    joined_gdf = joined_gdf.reset_index(drop=True)

    # aggregate to unique fid level
    grouped = joined_gdf.groupby('fid')
    agg_df = grouped.agg({'pop_weighted_psi': 'sum', 'pop2000': 'sum'}).reset_index()
    agg_df['pop_avg_psi'] = agg_df['pop_weighted_psi'] / agg_df['pop2000']

    avg_psi = grouped['psi'].mean().reset_index() # computing aggregated psi using the mean of all psi values within each country's geometry
    combined_df = agg_df.merge(avg_psi, on='fid', how='left')

    panel_gdf = panel_gdf.merge(combined_df, on='fid', how='left')
    panel_gdf = panel_gdf.drop(['pop_weighted_psi'], axis=1)

    # find where psi is null for a unique fid (these will be states that are too small, think island states) and compute the mean psi value for that fid
    all_null = panel_gdf.groupby('fid')['psi'].apply(lambda x: x.isna().all())
    fid_all_null = all_null[all_null].index.tolist()
    print("...Fid values where all psi observations are null:", fid_all_null)

    fid_all_null_geom = panel_gdf[panel_gdf['fid'].isin(fid_all_null)].groupby('fid', as_index=False).first()[['fid', 'geometry']]
    fid_all_null_geom.crs = "EPSG:4326"

    psi_help1_gdf = gpd.sjoin(fid_all_null_geom, psi_gdf, how='left', predicate='dwithin', distance=0.33) # find nearest psi value within 0.33 degrees
    psi_help1_gdf = psi_help1_gdf.groupby('fid', as_index=False).agg({'psi': 'mean'})
    psi_help1_gdf['pop_avg_psi'] = psi_help1_gdf['psi']

    print(psi_help1_gdf)

    psi_help2_gdf = panel_gdf.groupby('fid', as_index=False).first()[['fid', 'psi', 'pop_avg_psi']] # the psi values for all states that were not null
    psi_help2_gdf = psi_help2_gdf.dropna(subset=['psi']) # drop the fid values that had null psi values

    print(psi_help2_gdf)


    duplicates = set(psi_help1_gdf['fid']).intersection(set(psi_help2_gdf['fid']))
    if len(duplicates) == 0:
        # No duplicates—concatenate the DataFrames
        combined_df = pd.concat([psi_help1_gdf, psi_help2_gdf], ignore_index=True)
        print("...DataFrames appended successfully!")
    else:
        print(f"...Duplicate fid values found: {duplicates}")

    psi_mapping = combined_df.set_index('fid')['psi']
    panel_gdf['psi'] = panel_gdf['fid'].map(psi_mapping) # map psi values to the panel_gdf by fid
    pop_avg_psi_mapping = combined_df.set_index('fid')['pop_avg_psi']
    panel_gdf['pop_avg_psi'] = panel_gdf['fid'].map(pop_avg_psi_mapping) # map pop_avg_psi values to the panel_gdf by fid

    ######## C. COMPUTE CLIMATE INDEX (t and t-1 and t-2) AND MERGE W/ PANEL
    start_year  = int(panel_start_year - 5) # we compute one year previous so we can have a t-1 climate index column w/o loss of an observation
    end_year    = int(panel_end_year)

    annual_index = compute_annualized_index(clim_index, start_year, end_year)

    annual_index['cindex']
    annual_index['cindex_lag1y'] = annual_index['cindex'].shift(+1)
    annual_index['cindex_lag2y'] = annual_index['cindex'].shift(+2)
    annual_index['cindex_lag3y'] = annual_index['cindex'].shift(+3)
    annual_index = annual_index.rename(columns={'cindex': 'cindex_lag0y'})

    panel_gdf = panel_gdf.merge(annual_index, on='year', how='left')

    ######## E. POLISH PANEL
    cols = [col for col in panel_gdf.columns if col != 'geometry'] + ['geometry']
    panel_gdf = panel_gdf[cols]
    panel_gdf = panel_gdf.rename(columns={'cntry_n': 'country', 'fid': 'loc_id'})

    ###### F. TRANSFORM TO DESIRED RESPONSE VARIABLE: BINARY or COUNT
    if (response_var=='binary'):        # NEED TO MAKE THIS DYNAMIC FOR THE LAGGED TERMS!!!!
        panel_gdf['conflict_count'] = (panel_gdf['conflict_count'] > 0).astype(int)
        # final_gdf['conflict_count_lag1y'] = (final_gdf['conflict_count_lag1y'] > 0).astype(int)
        panel_gdf.rename(columns={'conflict_count': 'conflict_binary', 'conflict_count_lag1y': 'conflict_binary_lag1y'}, inplace=True)

    ######## PLOTTING
    if (plot_telecon==True):

        onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
        df = pd.read_csv(onset_path)    
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.onset_lon, df.onset_lat),
            crs="EPSG:4326"
        )

        last_obs = panel_gdf[panel_gdf['year'] == panel_end_year]
        # create a custom colormap
        import matplotlib.colors as mcolors
        bounds = [0, 1.80, np.max(last_obs['pop_avg_psi'])]
        cmap = mcolors.ListedColormap(["gainsboro", "blue"])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        var_in = 'pop_avg_psi'
        ## plot 1
        # Plot the geometries, coloring them by the psi value.
        ax = last_obs.plot(column=var_in,
                           cmap=cmap,
                           norm=norm,
                           legend=True,
                           figsize=(10, 6), 
                           edgecolor='black',
                           linewidth=0.25,
                           vmin=1.80)
        x, y = gdf['onset_lon'].values, gdf['onset_lat'].values
        ax.scatter(x, y, color='red', s=1.0, marker='o', zorder=5)
        plt.title(var_in)
        plt.show()

        ## plot 2
        median_val = last_obs[var_in].median()
        plt.hist(last_obs[var_in], bins='scott', color='red', edgecolor='black')
        plt.axvline(median_val, color='black', linestyle='dashed')
        x_center = (plt.xlim()[0] + plt.xlim()[1]) / 2; ymax = plt.ylim()[1]
        plt.text(x_center, ymax*0.9, f'Median: {median_val:.2f}', color='black', ha='center', va='center', backgroundcolor='white')
        plt.title(var_in)
        plt.show()


    cols = [col for col in panel_gdf.columns if col not in ['geometry', 'pop2000']]
    panel_gdf = panel_gdf[cols]

    return(panel_gdf)


panel = initalize_state_onset_panel(panel_start_year=1950,
                                    panel_end_year=2023,
                                    telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_DMI_FINAL.nc',
                                    pop_path = '/Users/tylerbagwell/Desktop/cccv_data/gpw-v4-population-count-rev11_totpop_15_min_nc/gpw_v4_population_count_rev11_15_min.nc',
                                    clim_index='dmi',
                                    response_var = 'binary',
                                    plot_telecon=True)
panel.to_csv('/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMI_new.csv', index=False)
# print(panel)
