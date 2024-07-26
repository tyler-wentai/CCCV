import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns


def mapper(geopoints1=None, geopoints2=None, include_bathymetry=True):
    # arg geopoints: geo-referenced points to be drawn on top of map
    # arg include_bathymetry: whether or not to draw ocean depth levels
    path_land = "data/map_packages/50m_cultural/ne_50m_admin_0_countries.shp"
    path_maritime_0 = "data/map_packages/ne_10m_bathymetry_L_0.shx"
    path_maritime_200 = "data/map_packages/ne_10m_bathymetry_K_200.shx"
    path_maritime_1000 = "data/map_packages/ne_10m_bathymetry_J_1000.shx"
    path_rivers = "data/map_packages/ne_50m_rivers_lake_centerlines.shx"

    gdf1 = gpd.read_file(path_land)
    gdf2 = gpd.read_file(path_maritime_0)
    gdf3 = gpd.read_file(path_maritime_200)
    gdf4 = gpd.read_file(path_maritime_1000)
    gdf5 = gpd.read_file(path_rivers)

    # construct plotting figure
    fig, ax = plt.subplots(figsize=(10, 6.6))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='white', alpha=0.5, zorder=1)

    gdf1.plot(ax=ax, edgecolor='gray', color='cornsilk', linewidth=0.5, zorder=2)
    gdf5.plot(ax=ax, edgecolor='#83c9f4', linewidth=0.5, zorder=2)
    if (include_bathymetry==True):
        gdf2.plot(ax=ax, edgecolor=None, color='#d9f0ff', zorder=0)
        gdf3.plot(ax=ax, edgecolor=None, color='#a3d5ff', zorder=0)
        gdf4.plot(ax=ax, edgecolor=None, color='#83c9f4', zorder=0)

    if (geopoints1 is not None):
        #sns.kdeplot(ax=ax, data=geopoints, x="longitude", y="latitude", bw_method=0.05, color='red', fill=True)
        geopoints1.plot(ax=ax, color='red', marker='o', markersize=0.1, label='Points', zorder=3)
    if (geopoints2 is not None):
        #sns.kdeplot(ax=ax, data=geopoints, x="longitude", y="latitude", bw_method=0.05, color='red', fill=True)
        geopoints2.plot(ax=ax, color='red', marker='o', markersize=0.1, label='Points', zorder=3)

    #ax.set_xlim([-20.0, 160.0])
    #ax.set_ylim([-40.0, +40.0])

    ax.set_xlim([-100.0, 160.0])
    ax.set_ylim([-45.0, +45.0])

    plt.tight_layout()
    #plt.savefig('map.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()



data_path = "data/pirate_attacks.csv"
df_points = pd.read_csv(data_path)
geometry = [Point(xy) for xy in zip(df_points['longitude'], df_points['latitude'])]
gdf_points2 = gpd.GeoDataFrame(df_points, geometry=geometry)


data_path = "/Users/tylerbagwell/Desktop/GEDEvent_v24_1.csv"
df_points = pd.read_csv(data_path)
geometry = [Point(xy) for xy in zip(df_points['longitude'], df_points['latitude'])]
gdf_points1 = gpd.GeoDataFrame(df_points, geometry=geometry)



mapper(gdf_points2)
