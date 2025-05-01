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
from oldcode.prepare_index import *
from utils.calc_annual_index import *
import cartopy.crs as ccrs
from shapely.geometry import Polygon

print('\n\nSTART ---------------------\n')

#
def calculate_hexagon_vertices(center_x, center_y, maximal_radius):
    """
    Calculates the vertices of a hexagon given the center coordinates and radius.

    :param center_x: X-coordinate of the center
    :param center_y: Y-coordinate of the center
    :param radius: Distance from the center to any vertex
    :return: List of (x, y) tuples representing the vertices
    """

    vertices = []
    for i in range(6):
        angle_deg = 60 * i #- 30  # Start with a flat side
        angle_rad = math.radians(angle_deg)
        x = center_x + maximal_radius * math.cos(angle_rad)
        y = center_y + maximal_radius * math.sin(angle_rad)
        vertices.append((x, y))
    return vertices


#
def create_grid(grid_polygon, localities, stepsize=1.0, show_grid=False):
    """
    Creates a square gridding over the specified region with specified stepsize in units of lat and lon degrees
    """
    africa_countries = ['Mauritania','Western Sahara','Ivory Coast','Niger','Guinea-Bissau','Tunisia','Equatorial Guinea','Malawi','Gabon','Liberia',
                        'Algeria','Lesotho','Sierra Leone','Mozambique','Ethiopia','Benin','Kenya','Guinea','Somalia','Madagascar',
                        'Morocco','Rwanda','South Sudan','Burkina Faso','Democratic Republic of the Congo','Botswana','Central African Republic','Nigeria',
                        'Mali','Namibia','Libya','Senegal','Burundi','eSwatini','Cameroon','United Republic of Tanzania','Togo','Republic of the Congo',
                        'Somaliland','Djibouti','Eritrea','Zimbabwe','Gambia','South Africa','Sudan','Zambia','Egypt',
                        'Chad','Angola','Uganda','Ghana'] #Africa has 54 reconized countries + 2 territories (Somaliland and Western Sahara) # removed Seychelles, Comoros, Mauritius, São Tomé and Principe, Cabo Verde
    asia_countries = ['India','Philippines','Myanmar','Nepal','Bangladesh','Tajikistan','Pakistan','Sri Lanka','Thailand','Indonesia','China','Vietnam',
                      'Afghanistan','Cambodia','Laos','Malaysia','Bhutan','Japan','Taiwan','South Korea','North Korea','Singapore','Turkmenistan','Uzbekistan',
                      'Kyrgyzstan','Iran','Papua New Guinea']
    south_america = ['Argentina','Uruguay','Chile','Brazil','Paraguay','Bolivia','Peru','Ecuador','Colombia','Venezuela','Guyana','Suriname','Panama',
                     'Nicaragua','Costa Rica','Honduras','El Salvador','Guatemala','Belize']
    
    remove_sovereignty = ['Antarctica']
    #remove_sovereignty = ['Antarctica','Monaco','Vatican City','San Marino','Taiwan','Western Sahara','Somaliland','Northern Cyprus','Marshall Islands','San Marino','Kashmir'] #used to determine median area of countries
    
    # Check that supplied grid_polygon is valid.
    allowed_polygons = ['square', 'hex', 'hexagon','country','Country','first_admin']
    if grid_polygon not in allowed_polygons:
        raise ValueError(f"Invalid grid_polygon '{grid_polygon}'. Allowed colors are: {allowed_polygons}.")

    # read in shp file data
    path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
    gdf1 = gpd.read_file(path_land)

    # grab the polygons related to each country (SOVEREIGNT) and 'explode' any countries
    # made of multipolygons into individual polygons
    if (localities=='Africa' or localities=='africa'):
        regions = africa_countries
    elif (localities=='Asia' or localities=='asia'):
        regions = asia_countries
    elif (localities=='South America' or localities=='south america'):
        regions = south_america
    elif (localities=='Global' or localities=='global'):
        regions = set(gdf1['SOVEREIGNT'])
    elif (localities=='NonAfrica' or localities=='nonafrica'):
        regions = set(gdf1['SOVEREIGNT']) - set(africa_countries)
    else:
        if not isinstance(localities, list):
            raise TypeError(f"'localities' argument should be a list if not a pre-specified region.")
        regions = localities

    gdf1 = gdf1[gdf1['SOVEREIGNT'].isin(regions)]
    gdf1 = gdf1[~gdf1['SOVEREIGNT'].isin(remove_sovereignty)]
    gdf1_help = gdf1.copy()
    gdf1 = gdf1.explode(index_parts=True)
    exploded_polygons = [gdf1.iloc[i].geometry for i in range(gdf1.shape[0])]

    gdf_countries = gdf1 # copy of gdf1 used later to find cell's dominant country.


    # Combine all individual polygons into a single multipolygon to ease the computation of the centroid,
    # centroid is used to form the grid around
    multi_polygon = shapely.geometry.MultiPolygon(exploded_polygons)
    gdf = gpd.GeoDataFrame(geometry=[multi_polygon])

    # compute region centroid and turn it into a gdf
    pnt = shapely.geometry.Point(multi_polygon.centroid.x, multi_polygon.centroid.y) 
    df = gpd.GeoDataFrame(geometry=[pnt], crs=4326)

    lon_center = df.geometry.iloc[0].x
    lat_center = df.geometry.iloc[0].y

    # compute the min and max lon and lat values of the entire region, determins spatial extent of grid
    lon_min = np.min(gdf.geometry.get_coordinates()['x']); lon_max = np.max(gdf.geometry.get_coordinates()['x'])
    lat_min = np.min(gdf.geometry.get_coordinates()['y']); lat_max = np.max(gdf.geometry.get_coordinates()['y'])

    # compute the grids lat and lon start and end values by 'ceiling' the 
    # above computed region's min and max lon and lat values
    lon_start   = lon_center - np.ceil(lon_center-lon_min)
    lon_end     = lon_center + np.ceil(lon_max-lon_center) + stepsize
    lat_start   = lat_center - np.ceil(lat_center-lat_min)
    lat_end     = lat_center + np.ceil(lat_max-lat_center) + stepsize

    if grid_polygon=='square':
        # compute coordinate mesh
        xcoords = np.arange(start=lon_start, stop=lon_end, step=stepsize)
        ycoords = np.arange(start=lat_start, stop=lat_end, step=stepsize)
        coords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1,2)

        # compute grid boxes' center points and create a GeoDataFrame of the grid
        centerpoints = gpd.points_from_xy(x=coords[:,0], y=coords[:,1])
        squares = [p.buffer(distance=(stepsize/2), cap_style=3) for p in centerpoints]
        df2 = gpd.GeoDataFrame(geometry=squares, crs=df.crs)
        # remove all grid boxes that do not contain a land regions
        gdf_final = df2[df2.intersects(gdf.geometry.iloc[0])]
    
    elif grid_polygon=='hex' or grid_polygon=='hexagon':
        # Not: the _stepsize_ argument in the function call refers to the _maximal radius_, R,
        # of the hexagons forming the grid mesh.

        # compute coordinate mesh
        dx_step = 2*(3*stepsize/2)
        dy_step = np.sqrt(3)*stepsize
        xcoords1 = np.arange(start=lon_start, stop=lon_end, step=dx_step)
        ycoords1 = np.arange(start=lat_start, stop=lat_end, step=dy_step)
        xcoords2 = np.arange(start=lon_start+(dx_step/2), stop=lon_end+(dx_step/2), step=dx_step)
        ycoords2 = np.arange(start=lat_start-(dy_step/2), stop=lat_end+(dy_step/2), step=dy_step)

        xcoords1 = xcoords1 #np.append(arr=xcoords1, values=xcoords2, axis=None)
        ycoords1 = ycoords1 #np.append(arr=ycoords1, values=ycoords2, axis=None)
        xcoords2 = xcoords2 #np.append(arr=xcoords1, values=xcoords2, axis=None)
        ycoords2 = ycoords2 #np.append(arr=ycoords1, values=ycoords2, axis=None)

        coords1 = np.array(np.meshgrid(xcoords1, ycoords1)).T.reshape(-1,2)
        coords2 = np.array(np.meshgrid(xcoords2, ycoords2)).T.reshape(-1,2)

        # compute grid boxes' center points and create a GeoDataFrame of the grid
        centerpoints1 = gpd.points_from_xy(x=coords1[:,0], y=coords1[:,1])
        centerpoints2 = gpd.points_from_xy(x=coords2[:,0], y=coords2[:,1])

        hexs1 = [shapely.geometry.Polygon(calculate_hexagon_vertices(center_x=p.x,
                                                                    center_y=p.y,
                                                                    maximal_radius=stepsize)) for p in centerpoints1]
        hexs1 = gpd.GeoDataFrame(geometry=hexs1, crs="EPSG:4326")

        hexs2 = [shapely.geometry.Polygon(calculate_hexagon_vertices(center_x=p.x,
                                                                    center_y=p.y,
                                                                    maximal_radius=stepsize)) for p in centerpoints2]
        hexs2 = gpd.GeoDataFrame(geometry=hexs2, crs="EPSG:4326")

        hexs_combined = pd.concat([hexs1, hexs2], ignore_index=True)
        hexs_gdf = gpd.GeoDataFrame(hexs_combined, crs=hexs1.crs)

        # remove all grid boxes that do not contain a land regions
        gdf_final = hexs_gdf[hexs_gdf.intersects(gdf.geometry.iloc[0])]

    elif grid_polygon=='country' or grid_polygon=='Country':
        gdf_final = gdf1_help['geometry']
        gdf_final = gdf_final.to_frame()

    elif grid_polygon=='first_admin':
        # Names conventions are different: "Guinea-Bissau" to "Guinea Bissau"; "South Sudan" to "S. Sudan"; "eSwatini" to "Swaziland".
        if (localities=='Africa' or localities=='africa' or localities=='Global' or localities=='global'):
            replacements = {
                "Guinea-Bissau": "Guinea Bissau",
                "South Sudan": "S. Sudan",
                "eSwatini": "Swaziland"
                }
            regions = [replacements.get(item, item) for item in regions]
        path_admin = "data/map_packages/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
        gdf_admin = gpd.read_file(path_admin)
        gdf_admin = gdf_admin[gdf_admin['admin'].isin(regions)]
        gdf_final = gdf_admin['geometry']
        gdf_final = gdf_final.to_frame()


    gdf_final.reset_index(inplace=True)
    gdf_final = gdf_final.drop('index', axis=1, inplace=False)
    gdf_final['loc_id'] = ['loc_'+str(i) for i in range(gdf_final.shape[0])]
    gdf_final = gdf_final.to_crs(4326)

    # determine the dominant country for each grid cell
    intersection_gdf = gpd.overlay(gdf_final, gdf_countries, how='intersection')
    if not intersection_gdf.crs.is_projected:
        intersection_gdf = intersection_gdf.to_crs(epsg=3857)  # Example: Web Mercator
    intersection_gdf['intersection_area'] = intersection_gdf.geometry.area
    idx = intersection_gdf.groupby('loc_id')['intersection_area'].idxmax()
    dominant_overlap = intersection_gdf.loc[idx][['loc_id', 'SOVEREIGNT']]
    grid_with_country = gdf_final.merge(dominant_overlap, on='loc_id', how='left')

    n1 = gdf_countries['SOVEREIGNT'].nunique()      # compute the starting number of unique countries
    n2 = grid_with_country['SOVEREIGNT'].nunique()  # compute the number of unique countries after finding each dominant country per cell

    countries_unique_to_list1 = set(gdf_countries['SOVEREIGNT']) - set(grid_with_country['SOVEREIGNT'])

    gdf_final = grid_with_country.copy()
    print(gdf_final.crs)

    # pring out results of country-cell matching
    print(f'Result of current gridsize produced a number of unique countries lost of {n1-n2}')
    print(f'   ·Starting number of unique countries:                                      {n1}')
    print(f'   ·Final number of unique countries after finding dominant country per cell: {n2}')
    if not countries_unique_to_list1:
        print(f'   ·Countries lost: NONE')
    else:
        print(f'   ·Countries lost: {countries_unique_to_list1}')

    # plotting
    if (show_grid==True):
        categorical_cmap = 'gist_ncar_r'#'Set1'
        ax = df.plot(color="violet", markersize=20, figsize=(6.5, 6.5), zorder=3) # plots the centroid of the entire region
        grid_with_country.plot(ax=ax, column='SOVEREIGNT', cmap=categorical_cmap,
                                       edgecolor='black', linewidth=0.75,  zorder=1)#, legend=True,
                                    #    legend_kwds={
                                    #        'loc': 'lower left',
                                    #        'ncols' : 2,
                                    #        'fontsize': 3.5,
                                    #        'markerscale' : 0.5
                                    #        })
        gdf1.plot(ax=ax, facecolor='none', zorder=2, edgecolor='k', linewidth=0.75)
        if (localities=='Global'):
            xticks = np.linspace(-180, 180, 5)
            yticks = np.linspace(-60, 60, 5)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels([f"{tick:.2f}" for tick in xticks])
            ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        # plt.savefig('/Users/tylerbagwell/Desktop/cccv_data/plots_results/global_grid_square4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        # Define a polygon with lat/lon coordinates
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
        gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # create a custom colormap
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        gl.top_labels       = False 
        ax.set_global()
        gdf_plot = gdf_final.plot(
            ax=ax, column='SOVEREIGNT', cmap=categorical_cmap,
            edgecolor='black',
            linewidth=0.75,
            zorder=1,
            transform=ccrs.PlateCarree())
        ax.add_geometries(gdf_final['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
        plt.title(r'$4^{\circ} \times 4^{\circ}$ gridding', fontsize=11)
        plt.tight_layout()
        plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_global_square4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return gdf_final


# 
def compute_weather_controls(start_year, end_year, polygons_gdf, annual_index):
    """
    Computes the annualized climate-index-signal-removed average temperature and precipitation for 
    each geometry contained in the polygons_gdf for years start_year to end_year.
    """
    print(" Computing weather controls...")

    # Function to convert longitude from 0-360 to -180 to 180
    def convert_longitude(ds):
        longitude = ds['longitude']
        longitude = ((longitude + 180) % 360) - 180
        ds = ds.assign_coords(longitude=longitude)
        return ds
    
    # Function to perform standardization of variables
    def standardize_group(x):
        std = x.std()
        if std == 0:
            return x * 0  # Assign zero or any other constant value
        else:
            return (x - x.mean()) / std
        
    # Function to perform de-trending
    def detrend_group_t2m(group):
        X = group['INDEX']
        y = group['t2m']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        group['t2m'] = model.resid
        return group
    
    # Function to perform de-trending
    def detrend_group_tp(group):
        X = group['INDEX']
        y = group['tp']
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        group['tp'] = model.resid
        return group

    # LOAD IN ANOMALIZED CLIMATE DATA
    # t2m: air temperature at 2 m
    # tp: total precipitation
    file_path_VAR1 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_t2m_raw.nc' # t2m
    file_path_VAR2 = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_tp_raw.nc'  # tp
    ds1 = xr.open_dataset(file_path_VAR1)
    ds2 = xr.open_dataset(file_path_VAR2)

    # change dates to time format:
    dates = pd.to_datetime(ds1['date'].astype(str), format='%Y%m%d')
    ds1 = ds1.assign_coords(date=dates)
    ds1 = ds1.rename({'date': 'time'})
    ds1 = ds1.rio.write_crs("EPSG:4326")                    # Ensure the dataset has a CRS
    ds1 = ds1.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))   # Select data within the specified year range

    dates = pd.to_datetime(ds2['date'].astype(str), format='%Y%m%d')
    ds2 = ds2.assign_coords(date=dates)
    ds2 = ds2.rename({'date': 'time'})
    ds2 = ds2.rio.write_crs("EPSG:4326")                    # Ensure the dataset has a CRS
    ds2 = ds2.sel(time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))   # Select data within the specified year range

    lon1 = ds1['longitude']
    lon2 = ds2['longitude']
    if lon1.max() > 180:
        ds1 = convert_longitude(ds1)
    if lon2.max() > 180:
        ds2 = convert_longitude(ds2)
    ds1 = ds1.sortby('longitude')
    ds2 = ds2.sortby('longitude')
    ds1 = ds1.assign_coords(
        longitude=np.round(ds1['longitude'], decimals=2),
        latitude=np.round(ds1['latitude'], decimals=2)
        )
    ds2 = ds2.assign_coords(
        longitude=np.round(ds2['longitude'], decimals=2),
        latitude=np.round(ds2['latitude'], decimals=2)
        )

    polygons_gdf = polygons_gdf.to_crs(ds1.rio.crs)         # Reproject polygons_gdf to match the dataset's CRS

    ds_yearly1 = ds1.groupby('time.year').mean()             # Group data by year and compute annual mean
    ds_yearly2 = ds2.groupby('time.year').mean()             # Group data by year and compute annual mean

    # Iterate over each polygon and compute the average temperature for each year
    # t2m
    results_list1 = []
    for idx, row in polygons_gdf.iterrows():
        print(f"t2m: processing polygon {idx+1}/{len(polygons_gdf)}")
        geometry = [mapping(row['geometry'])]

        ds_clipped = ds_yearly1.rio.clip(geometry, ds1.rio.crs, drop=False)  # Clip the dataset to the polygon

        mean_temp = ds_clipped['t2m'].mean(dim=('latitude', 'longitude'))   # Compute spatial mean over the clipped area

        df = mean_temp.to_dataframe().reset_index() # Convert to DataFrame
        df['loc_id'] = row['loc_id']  # Use the loc_id from polygons_gdf
        df = df.rename(columns={'year': 'tropical_year'})
        results_list1.append(df)

    results1 = pd.concat(results_list1, ignore_index=True)    # Concatenate all results into a single DataFrame
    results1 = results1.drop(columns=['number', 'spatial_ref'])

    results1 = results1.merge(annual_index, on='tropical_year', how='left')                 # merge climate index data
    results1 = results1.groupby('loc_id').apply(detrend_group_t2m).reset_index(drop=True)   # remove climate index signal via detrending
    results1.drop('INDEX', axis=1, inplace=True)                                            # drop climate index column
    results1['t2m'] = results1.groupby('loc_id')['t2m'].transform(standardize_group)        # standardize residuals over all years for each loc_id

    # tp
    results_list2 = []
    for idx, row in polygons_gdf.iterrows():
        print(f"tp: processing polygon {idx+1}/{len(polygons_gdf)}")
        geometry = [mapping(row['geometry'])]

        ds_clipped = ds_yearly2.rio.clip(geometry, ds2.rio.crs, drop=False)  # Clip the dataset to the polygon

        mean_temp = ds_clipped['tp'].mean(dim=('latitude', 'longitude'))   # Compute spatial mean over the clipped area

        df = mean_temp.to_dataframe().reset_index() # Convert to DataFrame
        df['loc_id'] = row['loc_id']  # Use the loc_id from polygons_gdf
        df = df.rename(columns={'year': 'tropical_year'})
        results_list2.append(df)

    results2 = pd.concat(results_list2, ignore_index=True)  # Concatenate all results into a single DataFrame
    results2 = results2.drop(columns=['number', 'spatial_ref'])

    results2 = results2.merge(annual_index, on='tropical_year', how='left')                 # merge climate index data
    results2 = results2.groupby('loc_id').apply(detrend_group_tp).reset_index(drop=True)    # remove climate index signal via detrending
    results2.drop('INDEX', axis=1, inplace=True)                                            # drop climate index column
    results2['tp'] = results2.groupby('loc_id')['tp'].transform(standardize_group)          # standardize residuals over all years for each loc_id

    #
    results = results1.merge(results2, on=['loc_id', 'tropical_year'], how='left')
    # print(results)

    return results

#
def prepare_gridded_panel_data(grid_polygon, localities, stepsize, nlag_cindex, nlag_conflict, clim_index, response_var='count', telecon_path=None, add_weather_controls=False, show_grid=False, show_gridded_aggregate=False):
    """
    Create a panel data set where each unit of analysis is an areal unit gridbox initialized 
    via the create_grid() function.
    """
    allowed_responses = ['binary', 'count']
    if response_var not in allowed_responses:
        raise ValueError(f"Invalid response var '{response_var}'. Allowed colors are: {allowed_responses}.")

    # create polygon grid
    polygons_gdf = create_grid(grid_polygon, localities=localities, stepsize=stepsize, show_grid=show_grid)

    # ensure CRS is WGS84
    if polygons_gdf.crs is None or polygons_gdf.crs.to_string() != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    # load conflict events dataset and convert to GeoDataFrame
    conflictdata_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
    conflict_df = pd.read_csv(conflictdata_path)
    conflict_gdf = gpd.GeoDataFrame(
        conflict_df,
        geometry=gpd.points_from_xy(conflict_df.onset_lon, conflict_df.onset_lat),
        crs='EPSG:4326'
        )

    # spatial join conflict events and polygon grid
    joined_gdf = gpd.sjoin(conflict_gdf, polygons_gdf, how='inner', predicate='within')

    # filter desired years! WILL CHANGE LETTER TO ALLOW FOR USER SPECIFIED YEARS
    desired_years = np.arange(np.min(conflict_df['year']), np.max(conflict_df['year'])+1)
    filtered_gdf = joined_gdf[joined_gdf['year'].isin(desired_years)]

    start_year  = np.min(desired_years)-nlag_cindex-1 #need the -1 because DEC(t-1)
    end_year    = np.max(desired_years) #+ 1

    annual_index = compute_annualized_index(clim_index, start_year, end_year)
    # annual_index['cindex'] = annual_index['cindex'] / annual_index['cindex'].std()
    print(annual_index)

    # group by polygon (loc_id) and year and then count number of conflicts for each grouping
    count_df = filtered_gdf.groupby(['loc_id', 'year']).size().reset_index(name='conflict_count')

    # create complete grid, necessary to also get 0 counts for polygon,year pairs with no conflicts
    polygon_ids = polygons_gdf['loc_id'].unique()
    years = desired_years
    complete_index = pd.MultiIndex.from_product([polygon_ids, years], names=['loc_id', 'year'])
    count_complete_df = count_df.set_index(['loc_id', 'year']).reindex(complete_index, fill_value=0).reset_index()

    # merge conflict counts back to polygons to retain geometry
    final_gdf = polygons_gdf[['loc_id', 'geometry', 'SOVEREIGNT']].merge(count_complete_df, on='loc_id', how='right')
    final_gdf = final_gdf[['loc_id', 'year', 'conflict_count', 'SOVEREIGNT', 'geometry']]

    ###### --- COMPUTE WEATHER (AIR TEMP and PRECIP) CONTROLS
    if add_weather_controls==True:
        weather_controls = compute_weather_controls((start_year+1), end_year, polygons_gdf, annual_index)

        for i in range(nlag_cindex+1):
            lag_string_var1 = 't2m_lag' + str(i) + 'y'
            weather_controls[lag_string_var1] = weather_controls['t2m'].shift(i)
        weather_controls['t2m_lagF1y'] = weather_controls['t2m'].shift(-1)
        weather_controls.drop('t2m', axis=1, inplace=True)
        for i in range(nlag_cindex+1):
            lag_string_var2 = 'tp_lag' + str(i) + 'y'
            weather_controls[lag_string_var2] = weather_controls['tp'].shift(i)
        weather_controls['tp_lagF1y'] = weather_controls['tp'].shift(-1)
        weather_controls.drop('tp', axis=1, inplace=True)
        # print(weather_controls)

        final_gdf = final_gdf.merge(weather_controls, on=['loc_id', 'year'], how='left')
    # print(final_gdf)

    ###### --- ADD OBSERVED ANNUALIZED CLIMATE INDEX VALUES TO PANEL
    for i in range(nlag_cindex+1):
        lag_string = 'cindex_lag' + str(i) + 'y'
        annual_index[lag_string] = annual_index['cindex'].shift(i)
    #annual_index['cindex_lagF1y'] = annual_index['cindex'].shift(-1) # Include one forward, i.e., future lag to test for spurious results
    annual_index.drop('cindex', axis=1, inplace=True)

    final_gdf = final_gdf.merge(annual_index, on='year', how='left')
    final_gdf = final_gdf.sort_values(['loc_id', 'year']) # ensure the shift operation aligns counts correctly for each loc_id in chronological order
    #final_gdf = final_gdf.dropna(subset=['cindex_lagF1y']) # need to remove NANs 

    # for i in range(nlag_conflict):
    #     lag_string = 'conflict_count_lag' + str(i+1) + 'y'
    #     final_gdf[lag_string] = final_gdf.groupby('loc_id')['conflict_count'].shift((i+1))
    #     final_gdf = final_gdf.dropna(subset=[lag_string])

    final_gdf.reset_index(drop=True, inplace=True)

    ###### --- COMPUTE AGGREGATED TELECONNECTION STRENGTH FOR EACH SPATIAL UNIT
    if telecon_path is not None:
        # Match all gridded psi values to a polygon via loc_id and then aggregate psi values
        # in each Polygon by taking the MAX psi value.

        def convert_longitude(ds):
            lon = ds['lon']
            lon = ((lon + 180) % 360) - 180
            ds = ds.assign_coords(lon=lon)
            return ds

        print('Computing gdf for psi...')
        psi = xr.open_dataarray(telecon_path)

        # Ensure that the DataArray has 'lat' and 'lon' coordinates
        if 'lat' not in psi.coords or 'lon' not in psi.coords:
            raise ValueError("DataArray must have 'lat' and 'lon' coordinates.")
        
        # Apply conversion if necessary
        lon1 = psi['lon']
        if lon1.max() > 180:
            psi = convert_longitude(psi)

        psi = psi.sortby('lon')
        lon1 = psi['lon']

        df_psi = psi.to_dataframe(name='psi').reset_index()
        df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['lon'], row['lat']), axis=1)
        psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
        psi_gdf = psi_gdf[['lat', 'lon', 'psi', 'geometry']]

        # check crs
        if psi_gdf.crs != polygons_gdf.crs:
            psi_gdf = psi_gdf.to_crs(polygons_gdf.crs)
            print("Reprojected gdf to match final_gdf CRS.")

        joined_gdf = gpd.sjoin(psi_gdf, polygons_gdf, how='left', predicate='within')

        cleaned_gdf = joined_gdf.dropna(subset=['loc_id'])
        cleaned_gdf = cleaned_gdf.reset_index(drop=True)

        grouped = joined_gdf.groupby('loc_id')
        mean_psi = grouped['psi'].mean().reset_index() # Computing aggregated psi using the MAX of all psis in polygon

        # for randomizing psi:
        # mean_psi['psi'] = np.random.permutation(mean_psi['psi']) # MAKE SURE TO COMMENT OUT!!!!!

        final_gdf = final_gdf.merge(mean_psi, on='loc_id', how='left')

    final_gdf = final_gdf.dropna(subset=['psi']) # remove all locations that do not have a psi value

    ###### --- TRANSFORM TO DESIRED RESPONSE VARIABLE: BINARY or COUNT
    if (response_var=='binary'):        # NEED TO MAKE THIS DYNAMIC FOR THE LAGGED TERMS!!!!
        final_gdf['conflict_count'] = (final_gdf['conflict_count'] > 0).astype(int)
        # final_gdf['conflict_count_lag1y'] = (final_gdf['conflict_count_lag1y'] > 0).astype(int)
        final_gdf.rename(columns={'conflict_count': 'conflict_binary', 'conflict_count_lag1y': 'conflict_binary_lag1y'}, inplace=True)

    # plot
    if (show_gridded_aggregate==True):
        # total_aggregate = final_gdf.groupby(['loc_id'])['conflict_binary'].sum().reset_index()
        # total_aggregate = polygons_gdf.merge(total_aggregate, left_on=['loc_id'], right_on=['loc_id'])
        # # total_aggregate = mean_psi
        # # total_aggregate = polygons_gdf.merge(total_aggregate, left_on=['loc_id'], right_on=['loc_id'])

        # # plotting
        # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # total_aggregate.plot(
        #     column='conflict_binary',    
        #     cmap='turbo',   #turbo    YlOrRd     PRGn
        #     legend=True,                   
        #     legend_kwds={'label': r"$\Psi$", 'orientation': "vertical", 'shrink': 0.6,
        #                  'ticks': [0, 5, 10, 15, 20, 25, 30]},
        #     ax=ax
        #     #vmax=500
        # )
        # colorbar = fig.axes[-1]
        # colorbar.set_ylabel(r"Count", rotation=90, fontsize=14)
        # polygons_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, vmin=0.5)
        # ax.set_title(r'No. of state-based conflict onsets (1950-2023)', fontsize=15)
        # ax.set_axis_off()
        # plt.savefig('/Users/tylerbagwell/Desktop/MAP_Global_onsetcount_hex.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.show()

        # plotting
        from matplotlib.colors import TwoSlopeNorm
        from matplotlib.gridspec import GridSpec
        import matplotlib as mpl
        psi_min = final_gdf['psi'].min()
        psi_max = final_gdf['psi'].max()

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        final_gdf.plot(
            column='psi',    
            cmap='Reds',   #turbo    YlOrRd     PRGn
            legend=True,                   
            legend_kwds={'orientation': "vertical", 'shrink': 0.6},
            ax=ax,
        )
        colorbar = fig.axes[-1]
        colorbar.set_ylabel(r"$\Psi$", rotation=0, fontsize=14)
        polygons_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, vmin=0.5)
        ax.set_title(r'Teleconnection strength, $\Psi$ (ENSO)', fontsize=15)
        ax.set_axis_off()
        plt.tight_layout()
        # plt.savefig('/Users/tylerbagwell/Desktop/MAP_Global_hex_NINO3_psi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        #####
        #####

        total_aggregate = final_gdf.groupby(['loc_id'])['psi'].mean().reset_index()
        total_aggregate = polygons_gdf.merge(total_aggregate, left_on=['loc_id'], right_on=['loc_id'])

        onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
        df = pd.read_csv(onset_path)    
        gdf = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.onset_lon, df.onset_lat),
            crs="EPSG:4326"
        )

        import cartopy.feature as cfeature
        import matplotlib.ticker as mticker
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
        # Define a polygon with lat/lon coordinates
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4)
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
        gl.ylocator = mticker.FixedLocator(range(-60, 91, 30))    # parallels every 30°
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # create a custom colormap
        import matplotlib.colors as mcolors
        import matplotlib.patches as mpatches
        bounds = [np.min(total_aggregate['psi']), 0, np.max(total_aggregate['psi'])] #psi_quants
        cmap = mcolors.ListedColormap(["gainsboro", "blue"])
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                              fill=None, edgecolor='green', linewidth=1.5,
                              transform=ccrs.PlateCarree())

        gl.top_labels       = False 
        ax.set_global()
        gdf_plot = total_aggregate.plot(
            column='psi',    
            cmap='PRGn', #cmap, #'PRGn',
            norm=TwoSlopeNorm(vmin=bounds[0], vcenter=bounds[1], vmax=bounds[2]),   
            legend=True,                   
            legend_kwds={
                'label': "Teleconnection percentile", 
                'orientation': "vertical", 
                'shrink': 0.6,
                #'ticks': bounds
            },
            ax=ax,
            transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
        )
        ax.add_geometries(polygons_gdf['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
        ax.coastlines()
        # ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)
        ax.add_patch(index_box)
        cbar = gdf_plot.get_figure().axes[-1]
        x, y = gdf['onset_lon'].values, gdf['onset_lat'].values
        ax.scatter(x, y, color='blue', s=1.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)
        # cbar.set_yticklabels(['0%', '90%', '100%'])
        plt.title('NINO3 Teleconnection', fontsize=11)
        plt.tight_layout()
        plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_NINO3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        ##
        fig, ax = plt.subplots(figsize=(8, 5)) # Adjust figsize as needed

        sns.histplot(data=total_aggregate, x='psi', bins='scott', kde=True, ax=ax)
        ax.set_ylabel("Frequency")

        plt.tight_layout()
        # plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/psi_hist_NINO3_square3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        # fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [2, 1]})

        # final_gdf.plot(
        #     column='psi',    
        #     cmap='Reds',   #turbo    YlOrRd     PRGn
        #     legend=True,                   
        #     legend_kwds={'label': r"$\Psi$", 'orientation': "vertical", 'shrink': 0.6,},
        #     ax=axes[0],
        # )
        # polygons_gdf.plot(ax=axes[0], edgecolor='black', facecolor='none', linewidth=0.2, vmin=0.5)
        # axes[0].set_title(r'Teleconnection strength, $\Psi$ (ANI)', fontsize=15)
        # axes[0].set_axis_off()
        # # plt.tight_layout()
        # # plt.savefig('/Users/tylerbagwell/Desktop/MAP_Global_psi_NINO3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # # plt.show()

        # sns.histplot(mean_psi['psi'], bins=30, stat='count', kde=False, color='gray', ax=axes[1])
        # axes[1].set_xlabel(r'Teleconnection strength, $\Psi$', fontsize=15)
        # axes[1].spines['top'].set_visible(False)
        # axes[1].spines['right'].set_visible(False)
        # axes[1].spines['bottom'].set_visible(True)
        # axes[1].spines['left'].set_visible(True)
        # plt.tight_layout()
        # # plt.savefig('/Users/tylerbagwell/Desktop/HIST_Asia_psi_ANI_country.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        # plt.show()

    # cols = [col for col in final_gdf.columns if col != 'geometry']
    # final_gdf = final_gdf[cols]

    return final_gdf


# 3.7225
# stepsize=3.5
panel = prepare_gridded_panel_data(grid_polygon='square', localities='Global', stepsize=4.0,
                                        nlag_cindex=3, nlag_conflict=0,
                                        clim_index = 'nino3',
                                        response_var='count',
                                        telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_mrsosNINO3.nc',
                                        add_weather_controls=False,
                                        show_grid=True, show_gridded_aggregate=True)
panel.to_csv('/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_mrsosNINO3_square4_wGeometry.csv', index=False)


