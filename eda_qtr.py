import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm

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
piracy_counts_qtr = df_piracy.resample('QE').size()     # aggregate to quarter level
piracy_counts_qtr.name = 'PIRACY_COUNTS'
    # loc1: gulf of guinea piracy
df_piracy_loc1 = df_piracy[df_piracy['nearest_country'].isin(['NGA','CMR','BEN','TGO','GHA','GAB'])]
piracy_counts_loc1_qtr = df_piracy_loc1.resample('QE').size()      # aggregate to quarter level
piracy_counts_loc1_qtr.name = 'PIRACY_COUNTS'
    # loc2: South East Asia
df_piracy_loc2 = df_piracy[df_piracy['nearest_country'].isin(['PHL','MYS','VNM','THA','IDN','MMR','SGP'])]
piracy_counts_loc2_qtr = df_piracy_loc2.resample('QE').size()      # aggregate to quarter level
piracy_counts_loc2_qtr.name = 'PIRACY_COUNTS'
    # loc3: Middle East/Gulf of Aden
df_piracy_loc3 = df_piracy[df_piracy['nearest_country'].isin(['SDN','EGY','YEM','DJI','SAU','SOM','DJI','OMN'])]
piracy_counts_loc3_qtr = df_piracy_loc3.resample('QE').size()      # aggregate to quarter level
piracy_counts_loc3_qtr.name = 'PIRACY_COUNTS'
    # loc4: South America
#df_piracy_loc4 = df_piracy[df_piracy['nearest_country'].isin(['BRA','PER','ECU','COL','GUY','VEN','SUR'])]
df_piracy_loc4 = df_piracy[df_piracy['nearest_country'].isin(['BGD','IND','LKA'])]
piracy_counts_loc4_qtr = df_piracy_loc4.resample('QE').size()      # aggregate to quarter level
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
df_oni = df_oni['ANOM'].resample('QE').mean() # aggregate to quarter level

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
n_lag = 20
for i in range(n_lag):
    lag_string = 'ANOM_lag' + str(i+1) + 'q'
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


# plot
# fit = np.polyfit(piracy_oni['ANOM'],  piracy_oni['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# plt.scatter(piracy_oni['ANOM'], piracy_oni['PIRACY_COUNTS'], color='k', s=3)
# plt.plot(piracy_oni['ANOM'], fit_fn(piracy_oni['ANOM']), color='blue', linestyle='-')
# plt.show()

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

fit = np.polyfit(piracy_oni_loc4['ANOM'],  piracy_oni_loc4['PIRACY_COUNTS'], 1)
fit_fn = np.poly1d(fit)
plt.scatter(piracy_oni_loc4['ANOM'], piracy_oni_loc4['PIRACY_COUNTS'], color='k', s=3)
plt.plot(piracy_oni_loc4['ANOM'], fit_fn(piracy_oni_loc4['ANOM']), color='red', linestyle='-')
plt.show()

# Xmat = sm.add_constant(piracy_oni_loc2['ANOM_lag6q'])
# model = sm.OLS(piracy_oni_loc2["PIRACY_COUNTS"],Xmat).fit()
# print(model.summary())

# print(model.params['ANOM_lag6q'])
# print(model.conf_int().loc['ANOM_lag6q'])


coefs_results = pd.DataFrame(columns=['Est', 'CI_l', 'CI_u'])
for i in range(n_lag+1):
    lag_string = 'ANOM'
    if i!=0:
        lag_string = 'ANOM_lag' + str(i) + 'q'
    
    Xmat = sm.add_constant(piracy_oni[lag_string])
    model = sm.OLS(piracy_oni["PIRACY_COUNTS"],Xmat).fit()
    coef_est = model.params[lag_string]
    coef_CI  = model.conf_int().loc[lag_string]

    coefs_results.loc[len(coefs_results)] = [coef_est, coef_CI[0], coef_CI[1]]

l= np.arange(0, n_lag+1)
plt.plot(l, coefs_results['Est'], color='darkgreen', marker='o', markersize=3)
plt.fill_between(l, coefs_results['CI_l'], coefs_results['CI_u'], interpolate=True, color='lightgreen', alpha=0.5)
plt.hlines(0, color='k', xmin=min(l), xmax=max(l), linewidth=0.5)
plt.show()

# print(df_piracy.shape)
# print(df_piracy_loc1.shape)
# print(df_piracy_loc2.shape)
# print(df_piracy_loc3.shape)
# print(df_piracy_loc4.shape)

# country_counts = df_piracy['nearest_country'].value_counts()
# country_counts.to_csv('country_counts.txt', sep='\t', index=True)