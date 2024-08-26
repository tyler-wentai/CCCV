import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import datetime
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns

print('\n\nSTART ---------------------\n')

# file_path_POP = '/Users/tylerbagwell/Desktop/GriddedPopulationoftheWorld_data/gpw_v4_population_count_rev11_2005_15_min.asc'

#
def compute_country_aggregate(nc_file_path, aggregate):
    """
    Compute the country-level aggregate mean value from a lat,lon gridded NetCDF data 
    set of values and returns a pandas DataFrame with a column of country name and 
    the mean value computed for that country.
    """
    # Check if an .nc file
    if not nc_file_path.lower().endswith('.nc'):
        raise ValueError(f"File '{nc_file_path}' is not a .nc file")

    # Grab country POLYGONs
    file_path_COUNTRIES = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
    gdf_countries = gpd.read_file(file_path_COUNTRIES)

    # Read in values
    nc_dat = xr.open_dataarray(nc_file_path)
    nc_dat['lon'] = xr.where(nc_dat['lon'] > 180, nc_dat['lon'] - 360, nc_dat['lon']) # NEED TO CHECK IF THIS TRANSFORMATION IS NEEDED!
    nc_dat = nc_dat.sortby('lon')

    lon_grid, lat_grid = np.meshgrid(nc_dat.lon.values, nc_dat.lat.values)
    points = [Point(lon, lat) for lon, lat in zip(lon_grid.flatten(), lat_grid.flatten())]

    gdf = gpd.GeoDataFrame({
    'geometry': points,
    'value': nc_dat.values.flatten()
    }, crs="EPSG:4326") # NEED TO CHECK IF THIS CRS MAPPING IS APPROPRIATE!

    gdf_with_countries = gpd.sjoin(gdf, gdf_countries, how="left", predicate="within")
    gdf_with_countries = gdf_with_countries[gdf_with_countries['SOVEREIGNT'].notna()] # SOVEREIGNT is the column for countries
    if (aggregate=='mean'):
        df_country_aggregate = gdf_with_countries.groupby('SOVEREIGNT')['value'].mean().reset_index()
    elif (aggregate=='median'):
        df_country_aggregate = gdf_with_countries.groupby('SOVEREIGNT')['value'].median().reset_index()
    else:
        raise ValueError("Specified aggregate is not a valid aggregate argument.")

    return df_country_aggregate



df = compute_country_aggregate(nc_file_path='/Users/tylerbagwell/Desktop/cccv_data_local/psi_Hsiang2011_oni.nc',
                               aggregate = 'mean')

ax = sns.histplot(data=df, x='value', stat='density', bins=13)
ax.set_title(r'Country-level teleconnection strength, $\Psi_i^{ONI}$')
ax.set_xlabel(r'$\Psi_i^{ONI}$')
# plt.savefig('plots/hist_psi_amm.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()




#### ---- population weighting
## pop setup
# pop = rioxarray.open_rasterio(file_path_POP)

# lon_grid_pop, lat_grid_pop = np.meshgrid(pop.x.values, pop.y.values)
# points_pop = [Point(lon, lat) for lon, lat in zip(lon_grid_pop.flatten(), lat_grid_pop.flatten())]

# print(pop.y.values)

# gdf_pop = gpd.GeoDataFrame({
#     'geometry': points_pop,
#     'value': pop.values.flatten()
# }, crs="EPSG:4326")

# print(gdf_psi.crs)
# print(gdf_pop.crs)

# def compute_average_value(geometry, gdf, value_column, radius):
#     buffer = geometry.buffer(radius)                # Create a buffer around the point with the specified radius
#     neighbors = gdf[gdf.geometry.within(buffer)]    # Find all points in gdf that are within this buffer
#     if not neighbors.empty:                         # Calculate the average of the values in the specified column
#         return neighbors[value_column].mean()
#     else:
#         return None

# radius = 0.25*(1.1)  # Working in degrees lat and lon
# print(gdf_pop)
# gdf_psi['pop_average_value'] = gdf_psi.apply(lambda row: compute_average_value(row.geometry, gdf_pop, 'value', radius), axis=1)
# print("\n", gdf_psi)