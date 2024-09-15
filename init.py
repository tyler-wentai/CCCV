import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import warnings

print('\n\nSTART ---------------------\n')


#
def create_grid(regions, stepsize=1.0, show_fig=False):
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

    if (show_fig==True):
        df = df.to_crs(4326)
        df2 = df2.to_crs(4326)
        ax = df.plot(color="violet", markersize=20, figsize=(6.5, 6.5), zorder=3)
        df2.boundary.plot(ax=ax, zorder=2, color='black', linewidth=0.75)
        gdf1.plot(ax=ax, color='lightgray', zorder=0, edgecolor='k', linewidth=0.75)
        plt.show()

    return df2


#
def create_gridded_panel_data(regions, stepsize, show_fig=False, show_tot_counts_fig=False):
    """
    Create a panel data set where each unit of analysis is an areal unit gridbox initialized 
    via the create_grid() function.
    """
    # create polygon grid
    polygons_gdf = create_grid(regions=regions, stepsize=stepsize, show_fig=show_fig)

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

    if (show_tot_counts_fig==True):
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
        plt.show()

    return final_gdf

data = create_gridded_panel_data(regions='Africa', stepsize=1.0, show_fig=False, show_tot_counts_fig=False)
print(data)