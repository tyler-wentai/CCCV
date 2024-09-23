import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import warnings
from datetime import datetime

print('\n\nSTART ---------------------\n')

#
def prepare_NINO3(file_path, start_date, end_date):
    """
    Prepare NINO3 index data as pd.Data.Frame from Standard PSL Format (https://psl.noaa.gov/gcos_wgsp/Timeseries/Nino3/)
    start_date and end_date must be formatted as datetime(some_year, 1, 1, 0, 0, 0)
    """
    # Read in data files
    nino3 = pd.read_csv(file_path, sep='\s+', skiprows=1, skipfooter=7, header=None, engine='python')
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
def create_grid(regions, stepsize=1.0, show_grid=False):
    """
    Creates a square gridding over the specified region with specified stepsize in units of lat and lon degrees
    """
    africa_countries = ['Mauritania','Western Sahara','Ivory Coast','Niger','Guinea-Bissau','Tunisia','Equatorial Guinea','Malawi','Gabon','Liberia',
                        'Cabo Verde','Algeria','Lesotho','Sierra Leone','Mozambique','Ethiopia','Benin','Kenya','Guinea','Somalia','Madagascar','Comoros',
                        'Morocco','Rwanda','South Sudan','Burkina Faso','Democratic Republic of the Congo','Botswana','Central African Republic','Nigeria',
                        'Mali','Namibia','Libya','Senegal','Burundi','eSwatini','Cameroon','United Republic of Tanzania','Togo','Mauritius','Republic of the Congo',
                        'Somaliland','Seychelles','Djibouti','Eritrea','Zimbabwe','Gambia','South Africa','Sudan','São Tomé and Principe','Zambia','Egypt',
                        'Chad','Angola','Uganda','Ghana'] #Africa has 54 reconized countries + 2 territories (Somaliland and Western Sahara)

    # print(len(africa_countries))

    # read in shp file data
    path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
    gdf1 = gpd.read_file(path_land)

    # grab the polygons related to each country (SOVEREIGNT) and 'explode' any countries
    # made of multipolygons into individual polygons
    if (regions=='Africa' or regions=='africa'):
        regions = africa_countries
    elif (regions=='Global' or regions=='global'):
        regions = set(gdf1['SOVEREIGNT'])
    else:
        if not isinstance(regions, list):
            raise TypeError(f"'regions' argument should be a list if not a pre-specified region.")
        regions = regions

    gdf1 = gdf1[gdf1['SOVEREIGNT'].isin(regions)]
    gdf1 = gdf1.explode(index_parts=True)
    exploded_polygons = [gdf1.iloc[i].geometry for i in range(gdf1.shape[0])]

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

    # compute coordinate mesh
    xcoords = np.arange(start=lon_start, stop=lon_end, step=stepsize)
    ycoords = np.arange(start=lat_start, stop=lat_end, step=stepsize)
    coords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1,2)

    # compute grid boxes' center points and create a GeoDataFrame of the grid
    centerpoints = gpd.points_from_xy(x=coords[:,0], y=coords[:,1])
    squares = [p.buffer(distance=(stepsize/2), cap_style=3) for p in centerpoints]
    df2 = gpd.GeoDataFrame(geometry=squares, crs=df.crs)

    # remove all grid boxes that do not contain a land regions
    df2 = df2[df2.intersects(gdf.geometry.iloc[0])]
    df2.reset_index(inplace=True)
    df2.drop('index', axis=1, inplace=True)
    df2['loc_id'] = ['loc_'+str(i) for i in range(df2.shape[0])]

    if (show_grid==True):
        df = df.to_crs(4326)
        df2 = df2.to_crs(4326)
        ax = df.plot(color="violet", markersize=20, figsize=(6.5, 6.5), zorder=3)
        df2.boundary.plot(ax=ax, zorder=2, color='black', linewidth=0.75)
        gdf1.plot(ax=ax, color='lightgray', zorder=0, edgecolor='k', linewidth=0.75)
        # plt.savefig('/Users/tylerbagwell/Desktop/grid_ex3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return df2


#
def prepare_gridded_panel_data(regions, stepsize, num_lag, show_grid=False, show_gridded_tot_counts=False):
    """
    Create a panel data set where each unit of analysis is an areal unit gridbox initialized 
    via the create_grid() function.
    """
    # create polygon grid
    polygons_gdf = create_grid(regions=regions, stepsize=stepsize, show_grid=show_grid)

    # ensure CRS is WGS84
    if polygons_gdf.crs is None or polygons_gdf.crs.to_string() != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)

    # load conflict events dataset and convert to GeoDataFrame
    conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'
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
    final_gdf = polygons_gdf[['loc_id', 'geometry']].merge(count_complete_df, on='loc_id', how='right')
    final_gdf = final_gdf[['loc_id', 'year', 'conflict_count', 'geometry']]

    # Add the observed annualized climate index values to panel dataset
    start_year  = np.min(desired_years)-num_lag-1 #need the -1 because DEC(t-1)
    end_year    = np.max(desired_years)

    annual_index = compute_annualized_NINO3_index(start_year, end_year)

    for i in range(num_lag+1):
        lag_string = 'INDEX_lag' + str(i) + 'y'
        annual_index[lag_string]= annual_index['INDEX'].shift((i))
    annual_index.drop('INDEX', axis=1, inplace=True)

    final_gdf = final_gdf.merge(annual_index, on='year', how='left')
    

    # plot
    if (show_gridded_tot_counts==True):
        total_counts = final_gdf.groupby(['loc_id'])['conflict_count'].sum().reset_index()
        total_counts = polygons_gdf.merge(total_counts, left_on=['loc_id'], right_on=['loc_id'])

        # plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        total_counts.plot(
            column='conflict_count',    
            cmap='turbo',                  
            legend=True,                   
            legend_kwds={'label': "Total Event Counts", 'orientation': "vertical"},
            ax=ax,
            vmax=500
        )
        ax.set_title('Total Event Counts per Polygon', fontsize=15)
        ax.set_axis_off()
        # plt.savefig('/Users/tylerbagwell/Desktop/grid_ex3_counts.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return final_gdf

# data = prepare_gridded_panel_data(regions='Africa', stepsize=3.0, num_lag=1, show_grid=True, show_gridded_tot_counts=False)

# print(data.shape)
# data.to_csv('/Users/tylerbagwell/Desktop/prepared_conflict_data.csv')

# annual_index = compute_annualized_NINO3_index(1960, 2024)
# print(annual_index)