import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
import sys

# ONI: Oceanic Nino Index (from NOAA, https://www.ncei.noaa.gov/access/monitoring/enso/sst#oni)
# loc1 = Gulf of Guinea
# loc2 = South East Asia
# loc3 = Middle East/Gulf of Aden
# loc4 = South America

path1 = "data/NOAA_ONI_data.txt"
path2 = "data/pirate_attacks.csv"
df_oni      = pd.read_csv(path1, sep='\s+')
df_piracy   = pd.read_csv(path2)

# Piracy data_frame setup
df_piracy['date'] = pd.to_datetime(df_piracy['date'])
df_piracy.set_index('date', inplace=True)               # index rows by date
piracy_counts_qtr = df_piracy.resample('YE').size()     # aggregate to quarter level
piracy_counts_qtr.name = 'PIRACY_COUNTS'
    # loc1: gulf of guinea piracy
df_piracy_loc1 = df_piracy[df_piracy['nearest_country'].isin(['NGA','CMR','BEN','TGO','GHA','GAB'])]
piracy_counts_loc1_qtr = df_piracy_loc1.resample('YE').size()      # aggregate to quarter level
piracy_counts_loc1_qtr.name = 'PIRACY_COUNTS'
    # loc2: South East Asia
df_piracy_loc2 = df_piracy[df_piracy['nearest_country'].isin(['PHL','MYS','VNM','THA','IDN','MMR','SGP'])]
piracy_counts_loc2_qtr = df_piracy_loc2.resample('YE').size()      # aggregate to quarter level
piracy_counts_loc2_qtr.name = 'PIRACY_COUNTS'
    # loc3: Middle East/Gulf of Aden
df_piracy_loc3 = df_piracy[df_piracy['nearest_country'].isin(['SDN','EGY','YEM','DJI','SAU','SOM','DJI','OMN'])]
piracy_counts_loc3_qtr = df_piracy_loc3.resample('YE').size()      # aggregate to quarter level
piracy_counts_loc3_qtr.name = 'PIRACY_COUNTS'
    # loc4: South America
#df_piracy_loc4 = df_piracy[df_piracy['nearest_country'].isin(['BRA','PER','ECU','COL','GUY','VEN','SUR'])]
df_piracy_loc4 = df_piracy[df_piracy['nearest_country'].isin(['BGD','IND','LKA'])]
piracy_counts_loc4_qtr = df_piracy_loc4.resample('YE').size()      # aggregate to quarter level
piracy_counts_loc4_qtr.name = 'PIRACY_COUNTS'


# ONI data_frame setup
season_to_months = {
    'NDJ': '01',
    'DJF': '02',
    'JFM': '03',
    'FMA': '04',
    'MAM': '05',
    'AMJ': '06',
    'MJJ': '07',
    'JJA': '08',
    'JAS': '09',
    'ASO': '10',
    'SON': '11',
    'OND': '12'
}
df_oni['MN'] = df_oni['SEAS'].map(season_to_months)
df_oni['date'] = pd.to_datetime(df_oni['YR'].astype(str) + '-' + df_oni['MN'] + '-01')
df_oni['date'] = pd.to_datetime(df_oni['date'])
df_oni.set_index('date', inplace=True)
df_oni = df_oni.sort_index()
df_oni = df_oni['ANOM'].resample('YE').mean() # aggregate to quarter level


# combine the two data sets
    # global
piracy_oni = pd.concat([df_oni, piracy_counts_qtr], axis=1)
piracy_oni = piracy_oni.dropna()
    # loc1
piracy_oni_loc1 = pd.concat([df_oni, piracy_counts_loc1_qtr], axis=1)
piracy_oni_loc1 = piracy_oni_loc1.dropna()
    # loc1
piracy_oni_loc2 = pd.concat([df_oni, piracy_counts_loc2_qtr], axis=1)
piracy_oni_loc2 = piracy_oni_loc2.dropna()
    # loc3
piracy_oni_loc3 = pd.concat([df_oni, piracy_counts_loc3_qtr], axis=1)
piracy_oni_loc3 = piracy_oni_loc3.dropna()
    # loc4
piracy_oni_loc4 = pd.concat([df_oni, piracy_counts_loc4_qtr], axis=1)
piracy_oni_loc4 = piracy_oni_loc4.dropna()

# create lagged columns of ONI anom.
n_lag = 4
for i in range(n_lag):
    lag_string = 'ANOM_lag' + str(i+1) + 'y'
    piracy_oni[lag_string]= piracy_oni['ANOM'].shift((i+1))
    piracy_oni_loc1[lag_string]= piracy_oni_loc1['ANOM'].shift((i+1))
    piracy_oni_loc2[lag_string]= piracy_oni_loc2['ANOM'].shift((i+1))
    piracy_oni_loc3[lag_string]= piracy_oni_loc3['ANOM'].shift((i+1))
    piracy_oni_loc4[lag_string]= piracy_oni_loc4['ANOM'].shift((i+1))

piracy_oni = piracy_oni.dropna()
piracy_oni_loc1 = piracy_oni_loc1.dropna()
piracy_oni_loc2 = piracy_oni_loc2.dropna()
piracy_oni_loc3 = piracy_oni_loc3.dropna()
piracy_oni_loc4 = piracy_oni_loc4.dropna()



# print(pearsonr(piracy_oni['ANOM'],piracy_oni['ANOM_lag1y']))
# print(pearsonr(piracy_oni['ANOM'],piracy_oni['ANOM_lag2y']))
# print(pearsonr(piracy_oni['ANOM'],piracy_oni['ANOM_lag3y']))
# print(pearsonr(piracy_oni['ANOM'],piracy_oni['ANOM_lag4y']))
# print(pearsonr(piracy_oni['ANOM'],piracy_oni['ANOM_lag5y']))


piracy_oni_loc1['YEAR'] = piracy_oni_loc1.index.year.astype(int)
piracy_oni_loc2['YEAR'] = piracy_oni_loc2.index.year.astype(int)
piracy_oni_loc3['YEAR'] = piracy_oni_loc3.index.year.astype(int)
piracy_oni_loc4['YEAR'] = piracy_oni_loc4.index.year.astype(int)

piracy_oni_loc1['LOC'] = 'loc1'
piracy_oni_loc2['LOC'] = 'loc2'
piracy_oni_loc3['LOC'] = 'loc3'
piracy_oni_loc4['LOC'] = 'loc4'

dfs = [piracy_oni_loc1, piracy_oni_loc2, piracy_oni_loc3, piracy_oni_loc4]
data_oni = pd.concat(dfs, axis=0, ignore_index=True)

cols = list(data_oni.columns) 
cols[0], cols[1] = cols[1], cols[0]
data_oni = data_oni[cols]  # Reorder the DataFrame columns

# data_oni.to_csv('/Users/tylerbagwell/Desktop/data_oni.csv', index=False)

from linearmodels.panel import PanelOLS


df_dummies = pd.get_dummies(data_oni['LOC'], drop_first=True).astype(int)

print(df_dummies)

# Add the dummy variables to the main DataFrame
df = pd.concat([data_oni, df_dummies], axis=1)


X = sm.add_constant(df[['ANOM','ANOM_lag1y','ANOM_lag2y','ANOM_lag3y','ANOM_lag4y','YEAR','loc2','loc3','loc4']])
y = df['PIRACY_COUNTS']

# sys.exit()
model = sm.OLS(y, X).fit()
print(model.summary())

sys.exit()






YEARS = data_oni['YEAR'] - np.min(data_oni['YEAR'])
data_oni = data_oni.set_index(['LOC','YEAR'])
print(data_oni)

data_oni['YEAR'] = data_oni.index.get_level_values('YEAR')

y = data_oni['PIRACY_COUNTS']
X = data_oni[['ANOM','ANOM_lag1y','ANOM_lag2y','ANOM_lag3y','YEAR']]
X = sm.add_constant(X)

model = PanelOLS(y, X, entity_effects=True)  # entity_effects=True applies unit fixed effects
result = model.fit()
print(result.summary)
print(result.estimated_effects)







sys.exit()

# plot
fit = np.polyfit(piracy_oni_loc4['ANOM'],  piracy_oni_loc4['PIRACY_COUNTS'], 1)
fit_fn = np.poly1d(fit)
plt.scatter(piracy_oni_loc4['ANOM'], piracy_oni_loc4['PIRACY_COUNTS'], color='k', s=3)
plt.plot(piracy_oni_loc4['ANOM'], fit_fn(piracy_oni_loc4['ANOM']), color='blue', linestyle='-')
plt.show()



# fit = np.polyfit(piracy_oni_loc1['ANOM'],  piracy_oni_loc1['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# plt.scatter(piracy_oni_loc1['ANOM'], piracy_oni_loc1['PIRACY_COUNTS'], color='k', s=3)
# plt.plot(piracy_oni_loc1['ANOM'], fit_fn(piracy_oni_loc1['ANOM']), color='green', linestyle='-')
# plt.show()

# fit = np.polyfit(piracy_oni_loc2['ANOM'],  piracy_oni_loc2['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# plt.scatter(piracy_oni_loc2['ANOM'], piracy_oni_loc2['PIRACY_COUNTS'], color='k', s=3)
# plt.plot(piracy_oni_loc2['ANOM'], fit_fn(piracy_oni_loc2['ANOM']), color='purple', linestyle='-')
# plt.show()

# fit = np.polyfit(piracy_oni_loc4['ANOM'],  piracy_oni_loc4['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# plt.scatter(piracy_oni_loc4['ANOM'], piracy_oni_loc4['PIRACY_COUNTS'], color='k', s=3)
# plt.plot(piracy_oni_loc4['ANOM'], fit_fn(piracy_oni_loc4['ANOM']), color='red', linestyle='-')
# plt.show()

# Xmat = sm.add_constant(piracy_oni_loc2['ANOM_lag6q'])
# model = sm.OLS(piracy_oni_loc2["PIRACY_COUNTS"],Xmat).fit()
# print(model.summary())

# print(model.params['ANOM_lag6q'])
# print(model.conf_int().loc['ANOM_lag6q'])


# coefs_results = pd.DataFrame(columns=['Est', 'CI_l', 'CI_u'])
# for i in range(n_lag+1):
#     lag_string = 'ANOM'
#     if i!=0:
#         lag_string = 'ANOM_lag' + str(i) + 'q'
    
#     Xmat = sm.add_constant(piracy_oni[lag_string])
#     model = sm.OLS(piracy_oni["PIRACY_COUNTS"],Xmat).fit()
#     coef_est = model.params[lag_string]
#     coef_CI  = model.conf_int().loc[lag_string]

#     coefs_results.loc[len(coefs_results)] = [coef_est, coef_CI[0], coef_CI[1]]

# l= np.arange(0, n_lag+1)
# plt.plot(l, coefs_results['Est'], color='darkgreen', marker='o', markersize=3)
# plt.fill_between(l, coefs_results['CI_l'], coefs_results['CI_u'], interpolate=True, color='lightgreen', alpha=0.5)
# plt.hlines(0, color='k', xmin=min(l), xmax=max(l), linewidth=0.5)
# plt.show()

# print(df_piracy.shape)
# print(df_piracy_loc1.shape)
# print(df_piracy_loc2.shape)
# print(df_piracy_loc3.shape)
# print(df_piracy_loc4.shape)

# country_counts = df_piracy['nearest_country'].value_counts()
# country_counts.to_csv('country_counts.txt', sep='\t', index=True)









###### DMI
from datetime import datetime


file_path_DMI = 'data/NOAA_DMI_data.txt' # DMI: Dipole Mode Index
start_date = datetime(1950, 1, 1, 0, 0, 0)
end_date = datetime(2023, 12, 1, 0, 0, 0)

# Read in data files
dmi = pd.read_csv(file_path_DMI, sep='\s+', skiprows=1, skipfooter=7, header=None, engine='python')
year_start = int(dmi.iloc[0,0])
dmi = dmi.iloc[:,1:dmi.shape[1]].values.flatten()
df_dmi = pd.DataFrame(dmi)
date_range = pd.date_range(start=f'{year_start}-01-01', periods=df_dmi.shape[0], freq='ME')
df_dmi.index = date_range
df_dmi.rename_axis('date', inplace=True)
df_dmi.columns = ['ANOM']

df_dmi = df_dmi.resample('QE').mean() # Aggregate to QUEARTERLY level


# combine the two data sets
    # global
piracy_dmi = pd.concat([df_dmi, piracy_counts_qtr], axis=1)
piracy_dmi = piracy_dmi.dropna()
    # loc1
piracy_dmi_loc1 = pd.concat([df_dmi, piracy_counts_loc1_qtr], axis=1)
piracy_dmi_loc1 = piracy_dmi_loc1.dropna()
    # loc1
piracy_dmi_loc2 = pd.concat([df_dmi, piracy_counts_loc2_qtr], axis=1)
piracy_dmi_loc2 = piracy_dmi_loc2.dropna()
    # loc3
piracy_dmi_loc3 = pd.concat([df_dmi, piracy_counts_loc3_qtr], axis=1)
piracy_dmi_loc3 = piracy_dmi_loc3.dropna()
    # loc4
piracy_dmi_loc4 = pd.concat([df_dmi, piracy_counts_loc4_qtr], axis=1)
piracy_dmi_loc4 = piracy_dmi_loc4.dropna()

# create lagged columns of ONI anom.
n_lag = 10
for i in range(n_lag):
    lag_string = 'ANOM_lag' + str(i+1) + 'q'
    piracy_dmi[lag_string]= piracy_dmi['ANOM'].shift((i+1))
    piracy_dmi_loc1[lag_string]= piracy_dmi_loc1['ANOM'].shift((i+1))
    piracy_dmi_loc2[lag_string]= piracy_dmi_loc2['ANOM'].shift((i+1))
    piracy_dmi_loc3[lag_string]= piracy_dmi_loc3['ANOM'].shift((i+1))
    piracy_dmi_loc4[lag_string]= piracy_dmi_loc4['ANOM'].shift((i+1))

piracy_dmi = piracy_dmi.dropna()
piracy_dmi_loc1 = piracy_dmi_loc1.dropna()
piracy_dmi_loc2 = piracy_dmi_loc2.dropna()
piracy_dmi_loc3 = piracy_dmi_loc3.dropna()
piracy_dmi_loc4 = piracy_dmi_loc4.dropna()


# fit = np.polyfit(piracy_dmi['ANOM'],  piracy_dmi['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# plt.scatter(piracy_dmi['ANOM'], piracy_dmi['PIRACY_COUNTS'], color='k', s=3)
# plt.plot(piracy_dmi['ANOM'], fit_fn(piracy_dmi['ANOM']), color='red', linestyle='-')
# plt.show()




coefs_results = pd.DataFrame(columns=['Est', 'CI_l', 'CI_u'])
for i in range(n_lag+1):
    lag_string = 'ANOM'
    if i!=0:
        lag_string = 'ANOM_lag' + str(i) + 'q'
    
    Xmat = sm.add_constant(piracy_dmi_loc4[lag_string])
    model = sm.OLS(piracy_dmi_loc4["PIRACY_COUNTS"],Xmat).fit()
    coef_est = model.params[lag_string]
    coef_CI  = model.conf_int().loc[lag_string]

    coefs_results.loc[len(coefs_results)] = [coef_est, coef_CI[0], coef_CI[1]]

l= np.arange(0, n_lag+1)
plt.plot(l, coefs_results['Est'], color='darkgreen', marker='o', markersize=3)
plt.fill_between(l, coefs_results['CI_l'], coefs_results['CI_u'], interpolate=True, color='lightgreen', alpha=0.5)
plt.hlines(0, color='k', xmin=min(l), xmax=max(l), linewidth=0.5)
plt.xlabel("L-lagged quarter")
plt.ylabel(r"$\beta$", rotation="horizontal")
plt.suptitle(r"Estimated marginal effect of quarterly lagged DMI via")
plt.text(0,5, "South America data")
plt.title(r" $ \text{piracy_count}_t = \alpha + \beta \text{DMI}_{t-L} + \epsilon_t$  (VERY NAIVE MODEL!!!)")
# plt.savefig('/Users/tylerbagwell/Desktop/DMI_effect_LOC4.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()
