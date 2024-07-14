import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def mapper(geopoints=None, include_bathymetry=True):
    # arg geopoints: geo-referenced points to be drawn on top of map
    # arg include_bathymetry: whether or not to draw ocean depth levels
    path_land = "/Users/tylerbagwell/Desktop/packages/Natural_Earth_quick_start/50m_cultural/ne_50m_admin_0_countries.shp"
    path_maritime_0 = "/Users/tylerbagwell/Desktop/packages/ne_10m_bathymetry_all/ne_10m_bathymetry_L_0.shp"
    path_maritime_200 = "/Users/tylerbagwell/Desktop/packages/ne_10m_bathymetry_all/ne_10m_bathymetry_K_200.shp"
    path_maritime_1000 = "/Users/tylerbagwell/Desktop/packages/ne_10m_bathymetry_all/ne_10m_bathymetry_J_1000.shp"

    gdf1 = gpd.read_file(path_land)
    gdf2 = gpd.read_file(path_maritime_0)
    gdf3 = gpd.read_file(path_maritime_200)
    gdf4 = gpd.read_file(path_maritime_1000)

    # construct plotting figure
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white', alpha=0.5, zorder=1)

    gdf1.plot(ax=ax, edgecolor='gray', color='lightgray', linewidth=0.5, zorder=2)
    if (include_bathymetry==True):
        gdf2.plot(ax=ax, edgecolor=None, color='#d9f0ff', zorder=0)
        gdf3.plot(ax=ax, edgecolor=None, color='#a3d5ff', zorder=0)
        gdf4.plot(ax=ax, edgecolor=None, color='#83c9f4', zorder=0)

    if (geopoints is not None):
        geopoints.plot(ax=ax, color='red', marker='o', markersize=1, label='Points')

    #ax.set_xlim([-20.0, 170.0])
    #ax.set_ylim([-50.0, +50.0])

    #ax.set_xlim([+90.0, 130.0])
    #ax.set_ylim([-30.0, +50.0])
    plt.tight_layout()
    #plt.savefig('map.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



data_path = "/Users/tylerbagwell/Documents/github/CCCV/data/pirate_attacks.csv"
df_points = pd.read_csv(data_path)
geometry = [Point(xy) for xy in zip(df_points['longitude'], df_points['latitude'])]
gdf_points = gpd.GeoDataFrame(df_points, geometry=geometry)



mapper(gdf_points)
