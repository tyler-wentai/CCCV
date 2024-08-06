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

path1 = "data/NOAA_ONI_data.txt"
path2 = "data/pirate_attacks.csv"
df_oni      = pd.read_csv(path1, sep='\s+')
df_piracy   = pd.read_csv(path2)

# Piracy data_frame setup
df_piracy['date'] = pd.to_datetime(df_piracy['date'])
df_piracy.set_index('date', inplace=True)               # index rows by date
piracy_counts_mn = df_piracy.resample('ME').size()      # aggregate by month
piracy_counts_mn.index = piracy_counts_mn.index + pd.offsets.MonthBegin(-1) # each month starts at day 1, note that this is just to be in sync with the oni data_frame
    # loc1: gulf of guinea piracy
df_piracy_loc1 = df_piracy[df_piracy['nearest_country'].isin(['NGA','CMR','BEN','TGO','GHA','GAB'])]
piracy_counts_loc1_mn = df_piracy_loc1.resample('ME').size()      # aggregate by month
piracy_counts_loc1_mn.index = piracy_counts_loc1_mn.index + pd.offsets.MonthBegin(-1) # each month starts at day 1, note that this is just to be in sync with the oni data_frame
    # loc2: South East Asia
df_piracy_loc2 = df_piracy[df_piracy['nearest_country'].isin(['PHL','MYS','VNM','THA','IDN','MMR','SGP'])]
piracy_counts_loc2_mn = df_piracy_loc2.resample('ME').size()      # aggregate by month
piracy_counts_loc2_mn.index = piracy_counts_loc2_mn.index + pd.offsets.MonthBegin(-1) # each month starts at day 1, note that this is just to be in sync with the oni data_frame
    # loc3: Middle East/Gulf of Aden
df_piracy_loc3 = df_piracy[df_piracy['nearest_country'].isin(['SDN','EGY','YEM','DJI','SAU','SOM','DJI','OMN'])]
piracy_counts_loc3_mn = df_piracy_loc3.resample('ME').size()      # aggregate by month
piracy_counts_loc3_mn.index = piracy_counts_loc3_mn.index + pd.offsets.MonthBegin(-1) # each month starts at day 1, note that this is just to be in sync with the oni data_frame

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

# combine the two data sets
    # global
piracy_oni = df_oni
piracy_oni['PIRACY_COUNTS'] = piracy_counts_mn
piracy_oni = piracy_oni.dropna(subset=['PIRACY_COUNTS'])
piracy_oni = piracy_oni.drop(columns=['SEAS','YR','MN'])
    # loc1
piracy_oni_loc1 = df_oni.copy()
piracy_oni_loc1['PIRACY_COUNTS'] = piracy_counts_loc1_mn
piracy_oni_loc1 = piracy_oni_loc1.dropna(subset=['PIRACY_COUNTS'])
piracy_oni_loc1 = piracy_oni_loc1.drop(columns=['SEAS','YR','MN'])
    # loc2
piracy_oni_loc2 = df_oni.copy()
piracy_oni_loc2['PIRACY_COUNTS'] = piracy_counts_loc2_mn
piracy_oni_loc2 = piracy_oni_loc2.dropna(subset=['PIRACY_COUNTS'])
piracy_oni_loc2 = piracy_oni_loc2.drop(columns=['SEAS','YR','MN'])
    # loc3
piracy_oni_loc3 = df_oni.copy()
piracy_oni_loc3['PIRACY_COUNTS'] = piracy_counts_loc3_mn
piracy_oni_loc3 = piracy_oni_loc3.dropna(subset=['PIRACY_COUNTS'])
piracy_oni_loc3 = piracy_oni_loc3.drop(columns=['SEAS','YR','MN'])


# create lagged columns of ONI anom. up to 24 months
n_lag = 12
for i in range(n_lag):
    lag_string = 'ANOM_lag' + str(i+1) + 'm'
    piracy_oni[lag_string]= piracy_oni['ANOM'].shift((i+1))
    piracy_oni_loc1[lag_string]= piracy_oni_loc1['ANOM'].shift((i+1))
    piracy_oni_loc2[lag_string]= piracy_oni_loc2['ANOM'].shift((i+1))
    piracy_oni_loc3[lag_string]= piracy_oni_loc3['ANOM'].shift((i+1))

# plot histograms of counts above/below temp. anom_val for comparison
anom_val = 0
df_above = piracy_oni[piracy_oni['ANOM'] > (+anom_val)]
df_below = piracy_oni[piracy_oni['ANOM'] < (-anom_val)]

# plt.figure(figsize=(10, 6))
# plt.hist(df_above['PIRACY_COUNTS'], bins='scott', density=True, color='red', edgecolor="red",\
#          alpha=0.6, label=f'Temperature > +{anom_val}')
# plt.hist(df_below['PIRACY_COUNTS'], bins='scott', density=True, color='blue', edgecolor="blue",\
#          alpha=0.6, label=f'Temperature < -{anom_val}')
# plt.title('Histogram of Counts Based on Temperature')
# plt.xlabel('Counts')
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()


# plot median counts versus temp. anom_val for comparison
med_counts_above = []
med_counts_below = []
temps = np.linspace(0,1.0,36)
for i in range(len(temps)):
    anom_val = temps[i]
    df_above = piracy_oni[piracy_oni['ANOM'] > (+anom_val)]
    df_below = piracy_oni[piracy_oni['ANOM'] < (-anom_val)]
    med_counts_above.append(df_above['PIRACY_COUNTS'].mean())
    med_counts_below.append(df_below['PIRACY_COUNTS'].mean())

fit_a = np.polyfit(temps, med_counts_above, 1)
fit_b = np.polyfit(temps, med_counts_below, 1)
fit_fn_a = np.poly1d(fit_a)
fit_fn_b = np.poly1d(fit_b)

# plt.figure(figsize=(10, 6))
# plt.plot(temps, med_counts_above, color='red', label=f'Months w/ ONI >  + Abs. temp. anom.')
# plt.plot(temps, fit_fn_a(temps), color='red', linestyle='--')
# plt.plot(temps, med_counts_below, color='blue', label=f'Months w/ ONI < - Abs. temp. anom.')
# plt.plot(temps, fit_fn_b(temps), color='blue', linestyle='--')
# plt.title('All global piracy data')
# plt.xlabel('Abs. temp. anom.')
# plt.ylabel('Mean piracy counts / month')
# plt.legend()
# plt.grid(True)
# plt.show()




# scatter plot of anom. vs. piracy counts
# piracy_oni_p = piracy_oni[piracy_oni['ANOM']>=0.]
# piracy_oni_m = piracy_oni[piracy_oni['ANOM']<1.0]
# fit = np.polyfit(piracy_oni['ANOM'],  piracy_oni['PIRACY_COUNTS'], 1)
# fit_fn = np.poly1d(fit)
# fit_m = np.polyfit(piracy_oni_m['ANOM'],  piracy_oni_m['PIRACY_COUNTS'], 1)
# fit_fn_m = np.poly1d(fit_m)
# plt.figure(figsize=(10, 6))
# plt.scatter(piracy_oni['ANOM'], piracy_oni['PIRACY_COUNTS'], color='k')
# plt.plot(piracy_oni['ANOM'], fit_fn(piracy_oni['ANOM']), color='blue', linestyle='-')
# plt.plot(piracy_oni_m['ANOM'], fit_fn_m(piracy_oni_m['ANOM']), color='blue', linestyle='-.')
# plt.title('Global Piracy Data: Monthly Piracy Count vs. ONI')
# plt.xlabel('ONI')
# plt.ylabel('Monthly Piracy Count')
# plt.show()


# piracy_oni_p = piracy_oni[piracy_oni['ANOM']>=0.]
# piracy_oni_m = piracy_oni[piracy_oni['ANOM']<0.0]
# plt.figure(figsize=(10, 6))
# plt.hist(piracy_oni_p['PIRACY_COUNTS'], bins='scott', density=True, color='red', edgecolor="red",\
#          alpha=0.6, label=f'ONI ANOM >= 0.0 (El Nino)')
# plt.hist(piracy_oni_m['PIRACY_COUNTS'], bins='scott', density=True, color='blue', edgecolor="blue",\
#          alpha=0.6, label=f'ONI ANOM < 0.0 (La Nina)')
# plt.title('Global Piracy Data Histogram: El Nino vs La Nina')
# plt.xlabel('Monthly Piracy Count')
# plt.ylabel('Density')
# plt.legend()
# plt.show()


piracy_oni_cleaned = piracy_oni.dropna()
piracy_oni_loc1_cleaned = piracy_oni_loc1.dropna()
piracy_oni_loc2_cleaned = piracy_oni_loc2.dropna()
piracy_oni_loc3_cleaned = piracy_oni_loc3.dropna()

pear_r_vals = []
pear_r_vals_loc1 = []
pear_r_vals_loc2 = []
pear_r_vals_loc3 = []
for i in range(n_lag):
    if (i==0):
        pear_r = pearsonr(piracy_oni_cleaned['ANOM'], piracy_oni_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc1 = pearsonr(piracy_oni_loc1_cleaned['ANOM'], piracy_oni_loc1_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc2 = pearsonr(piracy_oni_loc2_cleaned['ANOM'], piracy_oni_loc2_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc3 = pearsonr(piracy_oni_loc3_cleaned['ANOM'], piracy_oni_loc3_cleaned['PIRACY_COUNTS'])[0]
    else:
        lag_string = 'ANOM_lag' + str(i) + 'm'
        pear_r = pearsonr(piracy_oni_cleaned[lag_string], piracy_oni_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc1 = pearsonr(piracy_oni_loc1_cleaned[lag_string], piracy_oni_loc1_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc2 = pearsonr(piracy_oni_loc2_cleaned[lag_string], piracy_oni_loc2_cleaned['PIRACY_COUNTS'])[0]
        pear_r_loc3 = pearsonr(piracy_oni_loc3_cleaned[lag_string], piracy_oni_loc3_cleaned['PIRACY_COUNTS'])[0]
    pear_r_vals.append(pear_r)
    pear_r_vals_loc1.append(pear_r_loc1)
    pear_r_vals_loc2.append(pear_r_loc2)
    pear_r_vals_loc3.append(pear_r_loc3)

# print(pear_r_vals)

### Note: ONI probably isn't stationary (over our time window), so using a lagged correlation here might not make sense.
# plt.figure(figsize=(10, 6))
# plt.plot(pear_r_vals, color='white', marker='.')
# plt.plot(pear_r_vals_loc1, color='blue', marker='.')
# plt.plot(pear_r_vals_loc2, color='green', marker='.')
# plt.plot(pear_r_vals_loc3, color='goldenrod', marker='.', linestyle='-')
# plt.hlines(0,colors='k', xmin=0, xmax=n_lag)
# plt.show()

# print(pear_r_vals_loc2)
#print(acf(piracy_oni['PIRACY_COUNTS']))

# plt.plot(piracy_oni["PIRACY_COUNTS"], color='blue')
# plt.show()

piracy_oni_m = piracy_oni_loc2_cleaned[piracy_oni_cleaned['ANOM']<4.5]
Xmat = sm.add_constant(piracy_oni_m["ANOM_lag12m"])
model = sm.OLS(piracy_oni_m["PIRACY_COUNTS"],Xmat).fit()
print(model.summary())

#print(piracy_oni_cleaned)
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag1m']))
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag2m']))
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag3m']))
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag4m']))
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag5m']))
# print(pearsonr(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag6m']))

# plt.scatter(piracy_oni_cleaned['ANOM'],piracy_oni_cleaned['ANOM_lag6m'], color='k')
# plt.show()