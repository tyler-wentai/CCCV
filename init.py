import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

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

    print(len(africa_countries))

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




import warnings

grid_gdf = create_grid(regions='Africa', stepsize=1.0, show_fig=False)

conflictdata_path = '/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv'

conflict_data = pd.read_csv(conflictdata_path)

years_df = pd.DataFrame({'year': list(set(conflict_data['year']))})

# Add a temporary key for cross join
grid_gdf['key'] = 1
years_df['key'] = 1
# Perform the merge (cross join)
dataset = grid_gdf.merge(years_df, on='key')
dataset.drop('key', axis=1, inplace=True)
grid_gdf.drop('key', axis=1, inplace=True)

events_gdf = gpd.GeoDataFrame(
    conflict_data,
    geometry=gpd.points_from_xy(conflict_data.longitude, conflict_data.latitude),
    crs="EPSG:4326"  # Assuming WGS84 Latitude/Longitude
)

if events_gdf.crs != grid_gdf.crs:
    warnings.warn("Warning: events_gdf and grid_gdf did not have the same crs!")
    grid_gdf = grid_gdf.to_crs(events_gdf.crs)



# Perform spatial join: Find which polygon each event falls into
# 'inner' join returns only matching records
events_with_polygons = gpd.sjoin(events_gdf, grid_gdf, how='inner', predicate='within')


# 'sjoin' adds columns from polygons_gdf, including 'polygon_id' and 'year_right'
# To avoid confusion, rename columns appropriately

# Rename columns to differentiate between event year and polygon year
events_with_polygons = events_with_polygons.rename(columns={'year_left': 'event_year', 'year_right': 'polygon_year'})

# print(events_with_polygons)


# Now, filter rows where event_year matches polygon_year
# events_with_matching_year = events_with_polygons[events_with_polygons['event_year'] == events_with_polygons['polygon_year']]

event_counts = events_with_polygons.groupby(['loc_id', 'year']).size().reset_index(name='event_count')

# print(event_counts)

event_counts = grid_gdf.merge(event_counts, left_on=['loc_id'], right_on=['loc_id'])

# print(grid_gdf.merge(event_counts, left_on=['loc_id'], right_on=['loc_id']))

total_counts = event_counts.groupby(['loc_id'])['event_count'].sum().reset_index()
total_counts = grid_gdf.merge(total_counts, left_on=['loc_id'], right_on=['loc_id'])

print(total_counts)

# Create a plot with a specified size
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot polygons colored by 'total_event_count'
total_counts.plot(
    column='event_count',    # Column to color by
    cmap='OrRd',                   # Color map (e.g., OrRd, Viridis)
    legend=True,                   # Show color legend
    legend_kwds={'label': "Total Event Counts", 'orientation': "vertical"},
    ax=ax
)

# Add a title
ax.set_title('Total Event Counts per Polygon', fontsize=15)

# Remove axis for better aesthetics
ax.set_axis_off()

# Display the plot
plt.show()
