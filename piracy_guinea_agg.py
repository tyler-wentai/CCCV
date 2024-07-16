### Guinea here refers to the GULF OF GUINEA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from mapping import mapper
import statsmodels.discrete.discrete_model as sm

guinea_center_point = np.array([3.01, 4.34])  # lat, long of center of gulf of guinea


radius = 8.0

data_path = "data/pirate_attacks.csv"
gog_df = pd.read_csv(data_path)

gog_df['distance_from_gog'] = np.sqrt( (gog_df['latitude'] - guinea_center_point[0])**2 + \
                                       (gog_df['longitude']- guinea_center_point[1])**2 )

gog_df = gog_df[gog_df['distance_from_gog'] <= radius]


gog_df['date'] = pd.to_datetime(gog_df['date'])

gog_df.set_index('date', inplace=True)

# Aggregate by month
yearly_counts = gog_df.resample('Y').size()
# print("Yearly counts:")
# print(yearly_counts)


cov_path = "data/WBG_gdp_and_pop_data_1980-2023.csv"
cov_df = pd.read_csv(cov_path)

#print(cov_df.columns)
cov_df = cov_df[cov_df['Country Name']=='Nigeria']
cov_df = cov_df[cov_df['Series Name']=='GDP per capita (current US$)']
# monthly_counts['GDP per capita, t'] = cov_df

cov_t_df = cov_df.iloc[0,17:-3]
cov_tm1_df = cov_df.iloc[0,16:-4]

#print(cov_t_df)
cov_t_df = cov_t_df.reset_index(drop=True)
cov_tm1_df = cov_tm1_df.reset_index(drop=True)
#print(cov_t_df.shape)

#print(cov_t_df)


#print(yearly_counts)
yearly_counts = yearly_counts.reset_index(drop=True)
#print(yearly_counts.shape)


df = pd.concat([yearly_counts, cov_t_df, cov_tm1_df], axis=1)

print(df)

#print(cov_df.iloc[0,14:-3])
# plt.figure(figsize=(10, 5))
# monthly_counts.plot(kind='hist', bins=range(0, monthly_counts.max() + 2), edgecolor='black')
# plt.title('Monthly Event Counts')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.grid(True)
# plt.show()


#print(gog_df['date'])


# geometry = [Point(xy) for xy in zip(gog_df['longitude'], gog_df['latitude'])]
# gdf_points = gpd.GeoDataFrame(gog_df, geometry=geometry)

# print(gog_df.shape)

# mapper(gdf_points)