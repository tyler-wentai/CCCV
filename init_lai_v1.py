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
from prepare_index import *
from utils.calc_annual_index import *

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
    
    # Check that supplied grid_polygon is valid.
    allowed_polygons = ['square', 'hex', 'hexagon','country','Country','first_admin', 'FIRST_ADMIN', 'SQUARE', 'HEX', 'HEXAGON']
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

    if grid_polygon=='square' or grid_polygon=='SQUARE':
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
    
    elif grid_polygon=='hex' or grid_polygon=='hexagon' or grid_polygon=='HEX' or grid_polygon=='HEXAGON':
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

    elif grid_polygon=='country' or grid_polygon=='Country' or grid_polygon=='COUNTRY':
        gdf_final = gdf1_help['geometry']
        gdf_final = gdf_final.to_frame()

    elif grid_polygon=='first_admin' or grid_polygon=='FIRST_ADMIN':
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
        # plt.savefig('/Users/tylerbagwell/Desktop/grid_with_dom_country_Asia.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()

    return gdf_final



# 
def compute_agg_lai(start_year, end_year, polygons_gdf, nskip):
    """
    Computes the annualized ans spatially averaged leaf area index average for 
    each geometry contained in the polygons_gdf for years start_year to end_year.
    """
    print("Compute averaged leaf area index...")

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

    # LOAD IN LEAF AREA INDEX DATA
    lai_path = '/Users/tylerbagwell/Desktop/raw_climate_data/ERA5_LeafAreaIndex_data.nc'
    ds = xr.open_dataset(lai_path)
    ds = ds.isel(latitude=slice(0, None, int(nskip)), longitude=slice(0, None, int(nskip)))
    ds = ds['lai_lv']

    # change dates to time format:
    ds = ds.assign_coords(valid_time=ds.valid_time.dt.floor('D'))
    ds = ds.rio.write_crs("EPSG:4326")                    # Ensure the dataset has a CRS
    ds = ds.sel(valid_time=slice(f"{start_year}-01-01", f"{end_year}-12-31"))   # Select data within the specified year range

    lon1 = ds['longitude']
    if lon1.max() > 180:
        ds = convert_longitude(ds)
    ds = ds.sortby('longitude')

    polygons_gdf = polygons_gdf.to_crs(ds.rio.crs)         # Reproject polygons_gdf to match the dataset's CRS
    ds_yearly = ds.groupby('valid_time.year').mean()             # Group data by year and compute annual mean

    # Iterate over each polygon and compute the average lai_lv for each year
    results_list = []
    for idx, row in polygons_gdf.iterrows():
        print(f"lai_lv: processing polygon {idx+1}/{len(polygons_gdf)}")
        geometry = [mapping(row['geometry'])]

        ds_clipped = ds_yearly.rio.clip(geometry, ds.rio.crs, drop=False)  # Clip the dataset to the polygon

        mean_temp = ds_clipped.mean(dim=('latitude', 'longitude'))   # Compute spatial mean over the clipped area

        df = mean_temp.to_dataframe().reset_index() # Convert to DataFrame
        df['loc_id'] = row['loc_id']  # Use the loc_id from polygons_gdf
        results_list.append(df)

    results = pd.concat(results_list, ignore_index=True)    # Concatenate all results into a single DataFrame
    results = results.drop(columns=['number', 'spatial_ref'])
    results['lai_lv'] = results.groupby('loc_id')['lai_lv'].transform(standardize_group)    # standardize residuals over all years for each loc_id

    return results



#
def gridded_panel_lai_data(grid_polygon, localities, stepsize, year_start, year_end, nlag_clim_index, clim_index, telecon_path=None, add_weather_controls=False, show_grid=False, show_gridded_aggregate=False):
    """
    Create a panel data set where each unit of analysis is an areal unit gridbox initialized 
    via the create_grid() function.
    """
    # create polygon grid
    polygons_gdf = create_grid(grid_polygon, localities=localities, stepsize=stepsize, show_grid=show_grid)

    # ensure CRS is WGS84
    if polygons_gdf.crs is None or polygons_gdf.crs.to_string() != 'EPSG:4326':
        polygons_gdf = polygons_gdf.to_crs(epsg=4326)


    ###### --- COMPUTE AGGREGATED TELECONNECTION STRENGTH FOR EACH SPATIAL UNIT
    if telecon_path is not None:
        # Match all gridded psi values to a polygon via loc_id and then aggregate psi values
        # in each Polygon by taking the MAX psi value.
        print('Computing gdf for psi...')
        psi = xr.open_dataarray(telecon_path)

        # Ensure that the DataArray has 'lat' and 'lon' coordinates
        if 'latitude' not in psi.coords or 'longitude' not in psi.coords:
            raise ValueError("DataArray must have 'latitude' and 'longitude' coordinates.")

        df_psi = psi.to_dataframe(name='psi').reset_index()
        df_psi['geometry'] = df_psi.apply(lambda row: shapely.geometry.Point(row['longitude'], row['latitude']), axis=1)
        psi_gdf = gpd.GeoDataFrame(df_psi, geometry='geometry', crs='EPSG:4326')
        psi_gdf = psi_gdf[['latitude', 'longitude', 'psi', 'geometry']]

        # check crs
        if psi_gdf.crs != polygons_gdf.crs:
            psi_gdf = psi_gdf.to_crs(polygons_gdf.crs)
            print("Reprojected gdf to match final_gdf CRS.")

        joined_gdf = gpd.sjoin(psi_gdf, polygons_gdf, how='left', predicate='within')

        cleaned_gdf = joined_gdf.dropna(subset=['loc_id'])
        cleaned_gdf = cleaned_gdf.reset_index(drop=True)

        grouped = joined_gdf.groupby('loc_id')
        mean_psi = grouped['psi'].mean().reset_index() # Computing aggregated psi using the MAX of all psis in polygon

        polygons_gdf = polygons_gdf.merge(mean_psi, on='loc_id', how='left')

    polygons_gdf = polygons_gdf.dropna(subset=['psi']) # remove all locations that do not have a psi value

    ##
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    polygons_gdf.plot(
        column='psi',    
        cmap='YlOrRd',   #turbo    YlOrRd     PRGn
        legend=True,                   
        legend_kwds={'orientation': "vertical", 'shrink': 0.6},
        ax=ax,
    )
    colorbar = fig.axes[-1]
    colorbar.set_ylabel(r"", rotation=90, fontsize=14)
    ax.set_title(r'', fontsize=15)
    ax.set_axis_off()
    plt.show()

    # filter desired years! WILL CHANGE LETTER TO ALLOW FOR USER SPECIFIED YEARS
    start_year  = year_start - nlag_clim_index - 1
    end_year    = year_end

    annual_index = compute_annualized_index(clim_index, start_year, end_year)
    annual_index['cindex'] = annual_index['cindex'] / annual_index['cindex'].std()

    # combine the polygons and annual index dataframes to create a panel data set

    for i in range(nlag_clim_index+1):
        lag_string = 'INDEX_lag' + str(i) + 'y'
        annual_index[lag_string] = annual_index['cindex'].shift(i)
    annual_index['INDEX_lagF1y'] = annual_index['cindex'].shift(-1) # Include one forward, i.e., future lag to test for spurious results
    annual_index.drop('cindex', axis=1, inplace=True)

    ###### --- ADD OBSERVED ANNUALIZED CLIMATE INDEX VALUES TO PANEL
    panel = polygons_gdf.merge(annual_index, how='cross')
    panel = panel.sort_values(['loc_id', 'year']) # ensure the shift operation aligns counts correctly for each loc_id in chronological order
    panel = panel.dropna(subset=['INDEX_lag0y','INDEX_lag1y','INDEX_lagF1y']) # need to remove NANs 

    ###### --- ADD ANNUALIZED LEAF AREA INDEX VALUES TO PANEL
    lai_agg_data = compute_agg_lai(start_year = np.min(panel['year']),
                                   end_year = np.max(panel['year']), 
                                   polygons_gdf = polygons_gdf,
                                   nskip = 10.0)
    
    panel = panel.merge(lai_agg_data, on=['loc_id', 'year'], how='left')

    return panel

    

lai_panel = gridded_panel_lai_data( grid_polygon = 'SQUARE', 
                                    localities = 'Africa', 
                                    stepsize = 6.0, 
                                    year_start = 1950, 
                                    year_end = 2023, 
                                    nlag_clim_index = 1, 
                                    clim_index = 'nino34',
                                    telecon_path = '/Users/tylerbagwell/Desktop/cccv_data/processed_teleconnections/psi_nino34_LAND_nskip3.0_19502023_12months.nc',
                                    show_grid=False)

lai_panel = lai_panel.drop(columns=['geometry'])
lai_panel.to_csv('/Users/tylerbagwell/Desktop/test_panel.csv', index=False)