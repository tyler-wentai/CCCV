import geopandas as gpd
import shapely
import numpy as np
import matplotlib.pyplot as plt
import sys

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

    if (show_fig==True):
        df = df.to_crs(4326)
        df2 = df2.to_crs(4326)
        ax = df.plot(color="violet", markersize=200, figsize=(6.5, 6.5), zorder=3)
        df2.boundary.plot(ax=ax, zorder=2, color='red')
        gdf1.plot(ax=ax, color='gray', zorder=0)
        plt.show()

    return df2

my_grid = create_grid(regions='Africa', stepsize=1.0, show_fig=True)