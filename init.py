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
def prepare_NINO3(file_path, start_date, end_date):
    """
    Prepare NINO3 index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nino3 = pd.read_csv(file_path, sep=r'\s+', skiprows=1, skipfooter=7, header=None, engine='python')
    year_start = int(nino3.iloc[0,0])
    nino3 = nino3.iloc[:,1:nino3.shape[1]].values.flatten()
    df_nino3 = pd.DataFrame(nino3)
    date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_nino3.shape[0], freq='MS')
    df_nino3.index = date_range
    df_nino3.rename_axis('date', inplace=True)
    df_nino3.columns = ['ANOM']

    start_ts_l = np.where(df_nino3.index == start_date)[0]
    end_ts_l = np.where(df_nino3.index == end_date)[0]
    # Test if index list is empty, i.e., start_date or end_date are outside time series range
    if not start_ts_l:
        raise ValueError("start_ts_l is empty, start_date is outside range of NINO3 index time series.")
    if not end_ts_l:
        raise ValueError("end_ts_l is empty, end_date is outside range of NINO3 index time series.")
    
    start_ts_ind = int(start_ts_l[0])
    end_ts_ind = int(int(end_ts_l[0])+1)

    df_nino3 = df_nino3.iloc[start_ts_ind:end_ts_ind]

    return df_nino3


#
def compute_annualized_NINO3_index(start_year, end_year, save_path=False):
    """
    Computes the annualized NINO3 index via the average of the index of DEC(t-1),JAN(t),FEB(t) based on
    the method of Callahan 2023
    """
    # load index data
    clim_ind = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                            start_date=datetime(start_year, 1, 1, 0, 0, 0),
                            end_date=datetime(end_year, 12, 1, 0, 0, 0))

    # Compute the year index value as the average of DEC(t-1),JAN(t),FEB(t).
    clim_ind.index = pd.to_datetime(clim_ind.index)     # Ensure 'date' to datetime and extract year & month
    clim_ind['year'] = clim_ind.index.year
    clim_ind['month'] = clim_ind.index.month

    dec_df = clim_ind[clim_ind['month'] == 12].copy() # prepare December data from previous year
    dec_df['year'] = dec_df['year'] + 1  # Shift to next year
    dec_df = dec_df[['year', 'ANOM']].rename(columns={'ANOM': 'DEC_ANOM'})

    jan_feb_df = clim_ind[clim_ind['month'].isin([1, 2])].copy() # prepare January and February data for current year
    jan     = jan_feb_df[jan_feb_df['month'] == 1][['year', 'ANOM']].rename(columns={'ANOM': 'JAN_ANOM'})
    feb     = jan_feb_df[jan_feb_df['month'] == 2][['year', 'ANOM']].rename(columns={'ANOM': 'FEB_ANOM'})

    yearly = pd.merge(dec_df, jan, on='year', how='inner') # merge December, January, and February data
    yearly = pd.merge(yearly, feb, on='year', how='inner') # merge December, January, and February data

    yearly['INDEX'] = yearly[['DEC_ANOM', 'JAN_ANOM', 'FEB_ANOM']].mean(axis=1) # Calculate the average DJF ANOM value
    index_DJF = yearly[['year', 'INDEX']].sort_values('year').reset_index(drop=True)

    if (save_path!=False):
        np.save(save_path, index_DJF)

    return index_DJF


#
def create_grid(grid_polygon, regions, stepsize=1.0, show_grid=False):
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
    
    # Check that supplied grid_polygon is valid.
    allowed_polygons = ['square', 'hex', 'hexagon']
    if grid_polygon not in allowed_polygons:
        raise ValueError(f"Invalid grid_polygon '{grid_polygon}'. Allowed colors are: {allowed_polygons}.")

    # read in shp file data
    path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
    gdf1 = gpd.read_file(path_land)

    # grab the polygons related to each country (SOVEREIGNT) and 'explode' any countries
    # made of multipolygons into individual polygons
    if (regions=='Africa' or regions=='africa'):
        regions = africa_countries
    elif (regions=='Asia' or regions=='asia'):
        regions = asia_countries
    elif (regions=='South America' or regions=='south america'):
        regions = south_america
    elif (regions=='Global' or regions=='global'):
        regions = set(gdf1['SOVEREIGNT'])
    else:
        if not isinstance(regions, list):
            raise TypeError(f"'regions' argument should be a list if not a pre-specified region.")
        regions = regions

    gdf1 = gdf1[gdf1['SOVEREIGNT'].isin(regions)]
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
    
    gdf_final.reset_index(inplace=True)
    gdf_final = gdf_final.drop('index', axis=1, inplace=False)
    gdf_final['loc_id'] = ['loc_'+str(i) for i in range(gdf_final.shape[0])]
    gdf_final = gdf_final.to_crs(4326)

    # # determine the neighbors of each grid cell
    # gdf_help = gdf_final.copy()
    # projected_crs = "EPSG:3395"
    # gdf_help.to_crs(projected_crs, inplace=True) 
    # gdf_help['buffer'] = gdf_help.geometry.buffer(100)

    # neighbors = gpd.sjoin(gdf_help, gdf_help, how='left', predicate='intersects')
    # neighbors = neighbors[neighbors['loc_id_left'] != neighbors['loc_id_right']] # Remove self-matches
    # neighbors_list = neighbors.groupby('loc_id_left')['loc_id_right'].apply(list).reset_index()
    # neighbors_list = neighbors_list.rename(columns={'loc_id_left': 'loc_id', 'loc_id_right': 'neighbors'})
    # gdf_help = gdf_help.merge(neighbors_list, on='loc_id', how='left')
    
    # gdf_help['neighbors'] = gdf_help['neighbors'].apply(lambda x: x if isinstance(x, list) else [])
    # max_neighbors = gdf_help['neighbors'].apply(len).max()
    # print(f"Maximum number of neighbors: {max_neighbors}")
    # for i in range(max_neighbors):
    #     gdf_help[f'neighbor_{i+1}'] = gdf_help['neighbors'].apply(lambda x: x[i] if i < len(x) else None)
    # gdf_help = gdf_help.drop(columns=['neighbors'])


    # if not gdf_final['loc_id'].is_unique:
    #     raise ValueError("The 'loc_id' column in gdf_final must be unique.")
    
    # neighbor_cols_list = [f'neighbor_{i}' for i in range(1, max_neighbors + 1)]
    # neighbor_cols_list = ['loc_id'] + neighbor_cols_list
    # print(neighbor_cols_list)
    # gdf_final = gdf_final.merge(gdf_help[neighbor_cols_list],
    #                             on='loc_id',
    #                             how='left'
    #                             )
    

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
        # plt.savefig('/Users/tylerbagwell/Desktop/grid_with_dom_country_AFRICA.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return gdf_final


#
def prepare_gridded_panel_data(grid_polygon, regions, stepsize, nlag_psi, nlag_conflict, response_var='count', telecon_path=None, show_grid=False, show_gridded_aggregate=False):
    """
    Create a panel data set where each unit of analysis is an areal unit gridbox initialized 
    via the create_grid() function.
    """
    allowed_responses = ['binary', 'count']
    if response_var not in allowed_responses:
        raise ValueError(f"Invalid response var '{response_var}'. Allowed colors are: {allowed_responses}.")

    # create polygon grid
    polygons_gdf = create_grid(grid_polygon, regions=regions, stepsize=stepsize, show_grid=show_grid)

    # ensure CRS is WGS84
    if polygons_gdf.crs is None or polygons_gdf.crs.to_string() != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    # load conflict events dataset and convert to GeoDataFrame
    conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1_CLEANED.csv'
    # conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1_CLEANED_lowdeaths.csv'
    conflict_df = pd.read_csv(conflictdata_path)
    conflict_gdf = gpd.GeoDataFrame(
        conflict_df,
        geometry=gpd.points_from_xy(conflict_df.longitude, conflict_df.latitude),
        crs='EPSG:4326'
        )

    # spatial join conflict events and polygon grid
    joined_gdf = gpd.sjoin(conflict_gdf, polygons_gdf, how='inner', predicate='within')

    # filter desired years! WILL CHANGE LETTER TO ALLOW FOR USER SPECIFIED YEARS
    desired_years = list(set(conflict_df['year']))
    filtered_gdf = joined_gdf[joined_gdf['year'].isin(desired_years)]

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

    # Add the observed annualized climate index values to panel dataset
    start_year  = np.min(desired_years)-nlag_psi-1 #need the -1 because DEC(t-1)
    end_year    = np.max(desired_years)

    annual_index = compute_annualized_NINO3_index(start_year, end_year)
    for i in range(nlag_psi+1):
        lag_string = 'INDEX_lag' + str(i) + 'y'
        annual_index[lag_string] = annual_index['INDEX'].shift((i))
    annual_index.drop('INDEX', axis=1, inplace=True)

    final_gdf = final_gdf.merge(annual_index, on='year', how='left')
    final_gdf = final_gdf.sort_values(['loc_id', 'year']) # ensure the shift operation aligns counts correctly for each loc_id in chronological order

    for i in range(nlag_conflict):
        lag_string = 'conflict_count_lag' + str(i+1) + 'y'
        final_gdf[lag_string] = final_gdf.groupby('loc_id')['conflict_count'].shift((i+1))
        final_gdf = final_gdf.dropna(subset=[lag_string])

    final_gdf.reset_index(drop=True, inplace=True)

    #
    if telecon_path is not None:
        # Match all gridded psi values to a polygon via loc_id and then aggregate psi values
        # in each Polygon by taking the MAX psi value.
        print('Computing gdf for psi...')
        psi = xr.open_dataarray(telecon_path)

        # Ensure that the DataArray has 'lat' and 'lon' coordinates
        if 'lat' not in psi.coords or 'lon' not in psi.coords:
            raise ValueError("DataArray must have 'lat' and 'lon' coordinates.")

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
        mean_psi = grouped['psi'].max().reset_index() # Computing aggregated psi using the MAX of all psis in polygon

        final_gdf = final_gdf.merge(mean_psi, on='loc_id', how='left')

    final_gdf = final_gdf.dropna(subset=['psi']) # remove all locations that do not have a psi value

    # transform do desired response variable
    if (response_var=='binary'):        # NEED TO MAKE THIS DYNAMIC FOR THE LAGGED TERMS!!!!
        final_gdf['conflict_count'] = (final_gdf['conflict_count'] > 0).astype(int)
        final_gdf['conflict_count_lag1y'] = (final_gdf['conflict_count_lag1y'] > 0).astype(int)
        final_gdf.rename(columns={'conflict_count': 'conflict_binary', 'conflict_count_lag1y': 'conflict_binary_lag1y'}, inplace=True)

    # plot
    if (show_gridded_aggregate==True):
        total_aggregate = final_gdf.groupby(['loc_id'])['conflict_binary'].sum().reset_index()
        total_aggregate = polygons_gdf.merge(total_aggregate, left_on=['loc_id'], right_on=['loc_id'])
        # total_aggregate = mean_psi
        # total_aggregate = polygons_gdf.merge(total_aggregate, left_on=['loc_id'], right_on=['loc_id'])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        final_gdf.plot(
            column='psi',    
            cmap='YlOrRd',   #turbo    YlOrRd           
            legend=True,                   
            legend_kwds={'label': "psi", 'orientation': "vertical"},
            ax=ax,
            #vmax=500
        )
        ax.set_title(r'Teleconnection strength, psi', fontsize=15)
        ax.set_axis_off()
        plt.savefig('/Users/tylerbagwell/Desktop/binarycounts_Africa_hexagon_psi.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

        sns.histplot(mean_psi['psi'], bins=40, stat='density', kde=True, color='r')
        plt.savefig('/Users/tylerbagwell/Desktop/psi_binarycounts_Africa_hexagon.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return final_gdf



### Hex stepsize = 0.620401 for an area of 1.0!!!

panel_data = prepare_gridded_panel_data(grid_polygon='hex', regions='Africa', stepsize=0.620401,
                                        nlag_psi=7, nlag_conflict=1,
                                        response_var='binary',
                                        telecon_path = '/Users/tylerbagwell/Desktop/psi_callahan_NINO3_0dot5_soilw.nc',
                                        show_grid=True, show_gridded_aggregate=True)
# panel_data.to_csv('/Users/tylerbagwell/Desktop/panel_data_SouthAmerica_binary.csv', index=False)
# print(panel_data)
# nan_mask = panel_data.isna()
# print(nan_mask)
# nan_count_per_column = panel_data.isna().sum()
# print(nan_count_per_column)

# grid_data = create_grid(grid_polygon='square', regions='Africa', stepsize=2.00, show_grid=False)
# pd.set_option('display.max_colwidth', None)
# print(grid_data)




# def plot_neighbors(gdf, target_loc_id):
#     """
#     Plots the target polygon and its neighbors.

#     Parameters:
#     - gdf: GeoDataFrame containing 'geometry', 'loc_id', and neighbor columns.
#     - target_loc_id: The loc_id of the target polygon.
#     """
#     # Ensure the GeoDataFrame has the necessary columns
#     required_columns = ['geometry', 'loc_id', 'neighbor_1', 'neighbor_2', 
#                         'neighbor_3', 'neighbor_4', 'neighbor_5', 'neighbor_6']
#     if not all(col in gdf.columns for col in required_columns):
#         raise ValueError(f"GeoDataFrame must contain columns: {required_columns}")

#     # 3. Identify the Target Polygon
#     target = gdf[gdf['loc_id'] == target_loc_id]
#     if target.empty:
#         print(f"loc_id '{target_loc_id}' not found in the GeoDataFrame.")
#         return

#     # 4. Retrieve Neighbor loc_ids, excluding None or NaN
#     neighbor_cols = ['neighbor_1', 'neighbor_2', 'neighbor_3', 
#                      'neighbor_4', 'neighbor_5', 'neighbor_6']
#     neighbors_ids = target[neighbor_cols].iloc[0].dropna().tolist()

#     if not neighbors_ids:
#         print(f"No neighbors found for loc_id '{target_loc_id}'.")
#         return

#     neighbors = gdf[gdf['loc_id'].isin(neighbors_ids)]

#     # 6. Plotting
#     fig, ax = plt.subplots(figsize=(10, 10))
#     gdf.plot(ax=ax, color='lightgrey', edgecolor='black', aspect=1)
#     target.plot(ax=ax, color='blue', edgecolor='black', label='Target')
#     neighbors.plot(ax=ax, color='red', edgecolor='black', label='Neighbors')
#     plt.title(f"Neighbors of loc_id '{target_loc_id}'")
#     plt.show()

# plot_neighbors(grid_data, 'loc_603')
