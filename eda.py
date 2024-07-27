import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ONI: Oceanic Nino Index (from NOAA, https://www.ncei.noaa.gov/access/monitoring/enso/sst#oni)

path1 = "data/NOAA_ONI_data.txt"
path2 = "data/pirate_attacks.csv"
df_oni      = pd.read_csv(path1, sep='\s+')
df_piracy   = pd.read_csv(path2)

# Piracy data_frame setup
df_piracy['date'] = pd.to_datetime(df_piracy['date'])
df_piracy.set_index('date', inplace=True)               # index rows by date
piracy_counts_mn = df_piracy.resample('ME').size()      # aggregate by month
piracy_counts_mn.index = piracy_counts_mn.index + pd.offsets.MonthBegin(-1) # each month starts at day 1, note that this is just to be in sync with the oni data_frame

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
piracy_oni = df_oni
piracy_oni['PIRACY_COUNTS'] = piracy_counts_mn
piracy_oni = piracy_oni.dropna(subset=['PIRACY_COUNTS'])
piracy_oni = piracy_oni.drop(columns=['SEAS','YR','MN'])

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

plt.figure(figsize=(10, 6))
plt.plot(temps, med_counts_above, color='red', label=f'Months w/ ONI >  + Abs. temp. anom.')
plt.plot(temps, fit_fn_a(temps), color='red', linestyle='--')
plt.plot(temps, med_counts_below, color='blue', label=f'Months w/ ONI < - Abs. temp. anom.')
plt.plot(temps, fit_fn_b(temps), color='blue', linestyle='--')
plt.title('All global piracy data')
plt.xlabel('Abs. temp. anom.')
plt.ylabel('Mean piracy counts / month')
plt.legend()
plt.grid(True)
plt.show()