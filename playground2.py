import geopandas as gpd
from shapely.wkt import loads
import matplotlib.pyplot as plt


wkt_polygon = "POLYGON ((-116.02345169188895 31.44753724064035, -116.02345169188895 26.44753724064035, -121.02345169188895 26.44753724064035, -121.02345169188895 31.44753724064035, -116.02345169188895 31.44753724064035))"
polygon = loads(wkt_polygon)
gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[polygon])


gdf.plot()
plt.show()