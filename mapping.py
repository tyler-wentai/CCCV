import matplotlib.pyplot as plt
import geopandas as gpd

def mapper(include_bathymetry=True):
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

    ax.grid(True, which='both', linestyle='--', linewidth=1.0, color='white', alpha=0.5, zorder=1)

    gdf1.plot(ax=ax, edgecolor='gray', color='lightgray', linewidth=0.5, zorder=2)
    if (include_bathymetry==True):
        gdf2.plot(ax=ax, edgecolor=None, color='#d9f0ff', zorder=0)
        gdf3.plot(ax=ax, edgecolor=None, color='#a3d5ff', zorder=0)
        gdf4.plot(ax=ax, edgecolor=None, color='#83c9f4', zorder=0)

    ax.set_xlim([-20.0, 170.0])
    ax.set_ylim([-50.0, +50.0])
    plt.show()

mapper()
