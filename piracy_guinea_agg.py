### Guinea here refers to the GULF OF GUINEA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from mapping import mapper
import statsmodels.api as sm

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
yearly_counts = gog_df.resample('YE').size()
# print("Yearly counts:")
# print(yearly_counts)


cov_path = "data/WBG_gdp_and_pop_data_1980-2023.csv"
cov_df = pd.read_csv(cov_path)

#print(cov_df.columns)
cov_df = cov_df[cov_df['Country Name']=='Nigeria']
cov_df1 = cov_df[cov_df['Series Name']=='GDP per capita (current US$)']
cov_df2 = cov_df[cov_df['Series Name']=='Population, total']

import_dat = pd.Series(data = {'1996':5335465, '1997':6363504, '1998':5764245, '1999':4482496, '2000':5817035,\
                               '2001':7958202, '2002':8758472, '2003':14891746, '2004':15158796, '2005':15425846,\
                               '2006':22903337, '2007':32357346, '2008':28193597, '2009':33906281, '2010':44235268,\
                               '2011':63971541, '2012':35872509, '2013':44598201, '2014':46532265, '2015':33830878,\
                               '2016':35194301, '2017':31270090, '2018':43011523, '2019':47369076, '2020':55455401})
# monthly_counts['GDP per capita, t'] = cov_df




cov1_t_df = cov_df1.iloc[0,20:-3]
cov1_tm1_df = cov_df1.iloc[0,19:-4]

cov2_t_df = cov_df2.iloc[0,20:-3]
cov2_tm1_df = cov_df2.iloc[0,19:-4]


#print(cov_t_df)
cov1_t_df = cov1_t_df.reset_index(drop=True)
cov1_tm1_df = cov1_tm1_df.reset_index(drop=True)

cov2_t_df = cov2_t_df.reset_index(drop=True)
cov2_tm1_df = cov2_tm1_df.reset_index(drop=True)

import_dat = import_dat.reset_index(drop=True)
#print(cov_t_df.shape)

#print(cov_t_df)


#print(yearly_counts)
yearly_counts = yearly_counts.reset_index(drop=True)
#print(yearly_counts.shape)


df = pd.concat([yearly_counts[0:-3], cov1_t_df, cov1_tm1_df, cov2_t_df, cov2_tm1_df, import_dat], axis=1)
df['const'] = 1

#print(df)


df.columns = ["y","gdp_t","gdp_tm1","pop_t","pop_tm1","import","const"]
df = df.map(lambda x: pd.to_numeric(x, errors='coerce'))

#print(type(df['x_t']))
df['gdp_t'] = df['gdp_t'].transform(np.log)
df['gdp_tm1'] = df['gdp_tm1'].transform(np.log)
df['pop_t'] = df['pop_t'].transform(np.log)
df['pop_tm1'] = df['pop_tm1'].transform(np.log)
df['import'] = df['import'].transform(np.log)
# print(df[['const','x_t']])

print(df.corr())

model = sm.GLM(df['y'], df[['const','gdp_t','pop_t']], family=sm.families.NegativeBinomial())
results = model.fit()

print(results.summary())




# Define the formula for the model
# formula = 'y ~ x_t'

# # Fit the model
# model = smf.negativebinomial(formula, data=df).fit()

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