import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_ccf
from calc_annual_index import *

print("\n")

start_year = 1949
end_year = 2024


nino34  = prepare_NINO34(file_path='data/NOAA_NINO34_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
nino34.columns = ["nino34"]

nino3   = prepare_NINO3(file_path='data/NOAA_NINO3_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
nino3.columns = ["nino3"]

dmi     = prepare_DMI(file_path = 'data/NOAA_DMI_data.txt',
                                start_date=datetime(start_year, 1, 1, 0, 0, 0),
                                end_date=datetime(end_year, 12, 1, 0, 0, 0))
dmi.columns = ["dmi"]

dat_mon = nino34.copy()
dat_mon["nino3"]    = nino3
dat_mon["dmi"]      = dmi

# print(dat_mon.corr())

# x = dat_mon["nino3"]
# y = dat_mon["dmi"]
# fig, ax = plt.subplots(figsize=(8,4))
# plot_ccf(x, y, ax=ax)
# ax.set_title("Cross-correlations")
# plt.show()

dat_help = dat_mon.copy()

nlag = 24
for lag in range(1, nlag + 1):
    dat_help[f"nino3_lag{lag}"] = dat_help["nino3"].shift(lag)
dat_help = dat_help.loc["1950-01-01":"2023-12-31"]


# perform the MLR
X = dat_help[["nino3"] + [f"nino3_lag{lag}" for lag in range(1, nlag + 1)]]
y = dat_help["dmi"]

X = sm.add_constant(X)
model = sm.OLS(y, X, missing="drop").fit()
# print(model.summary())

params = model.params
print(params)

enso_component = (
    params["nino3"] * dat_help["nino3"]
    + sum(params[f"nino3_lag{lag}"] * dat_help[f"nino3_lag{lag}"] for lag in range(1, nlag+1))
)
dat_help["dmi_noenso"] = dat_help["dmi"] - enso_component

# add dmi_noenso back to dat_mon
dat_mon['dmi_noenso'] = dat_help["dmi_noenso"]
dat_mon = dat_mon.loc["1950-01-01":"2023-12-31"]

# print(np.round(dat_mon.corr(),4))


#
# plt.figure(figsize=(12, 5))
# plt.plot(dat_help.index, dat_help["dmi"], label="dmi", linewidth=2, alpha=0.7)
# plt.plot(dat_help.index, dat_help["dmi_noenso"], label="dmi_noenso", linewidth=2, alpha=0.7)
# plt.xlabel("Time")
# plt.ylabel("Index value")
# plt.legend()
# plt.tight_layout()
# plt.show()

#
# x = dat_mon["nino3"]
# y = dat_mon["dmi_noenso"]
# fig, ax = plt.subplots(figsize=(8,4))
# plot_ccf(x, y, ax=ax)
# ax.set_title("Cross-correlations")
# plt.show()


##########################################################################################

# # annualized
nino34_ann = compute_annualized_index("nino34", start_year, end_year).set_index("year")
nino34_ann.columns = ["nino34"]

nino3_ann  = compute_annualized_index("nino3",  start_year, end_year).set_index("year")
nino3_ann.columns = ["nino3"]

dmi_ann    = compute_annualized_index("dmi",    start_year, end_year).set_index("year")
dmi_ann.columns = ["dmi"]

dat_ann = nino34_ann.copy()
dat_ann["nino3"] = nino3_ann
dat_ann["dmi"]   = dmi_ann


# dmi_noense
son = dat_mon.loc[dat_mon.index.month.isin([9, 10, 11]), "dmi_noenso"]
dmi_noenso_son = son.groupby(son.index.year).mean()
dmi_noenso_son.index.name = "year"

dat_ann["dmi_noenso"] = dmi_noenso_son
dat_ann = dat_ann.loc[1950:2023]

#
dat_ann = dat_ann.copy()
X = sm.add_constant(dat_ann["nino3"])   # predictor
y = dat_ann["dmi"]  
model = sm.OLS(y, X, missing="drop").fit()
dat_ann["dmi_noenso_ann"] = model.resid

# print(dat_ann)
print(dat_ann.corr())

# save
print(dat_ann["dmi_noenso_ann"])
dat_ann["dmi_noenso_ann"].to_csv("data/dmi_noenso_ann.csv", header=True)

#
# plt.figure(figsize=(12, 5))
# plt.plot(dat_ann.index, dat_ann["dmi"], label="dmi", linewidth=2, alpha=0.7, marker="o", markersize=4)
# plt.plot(dat_ann.index, dat_ann["dmi_noenso_ann"], label="dmi_noenso", linewidth=2, alpha=0.7,  marker="o", markersize=4)
# plt.xlabel("Time")
# plt.ylabel("Index value")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()



# x = dat_ann["nino3"]
# y = dat_ann["dmi_noenso_ann"]
# fig, ax = plt.subplots(figsize=(8,4))
# plot_ccf(x, y, ax=ax, lags=4)
# ax.set_title("Cross-correlations")
# plt.show()

sys.exit()


import pandas as pd
from scipy.stats import pearsonr

def corr_with_pvalues(df):
    cols = df.columns
    n = len(cols)
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for i in range(n):
        for j in range(n):
            r, p = pearsonr(df[cols[i]].dropna(), df[cols[j]].dropna())
            corr.iloc[i, j] = r
            pval.iloc[i, j] = p
    return corr, pval

corr, pval = corr_with_pvalues(dat_ann)

print("Correlations:")
print(corr)
print("\nP-values:")
print(np.round(pval,4))