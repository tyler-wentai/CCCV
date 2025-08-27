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


# # annualized
nino34_ann = compute_annualized_index("nino34", start_year, end_year).set_index("year")
nino34_ann.columns = ["nino34"]

nino3_ann  = compute_annualized_index("nino3",  start_year, end_year).set_index("year")
nino3_ann.columns = ["nino3"]

dmi_ann    = compute_annualized_index("dmi",    start_year, end_year).set_index("year")
dmi_ann.columns = ["dmi"]

dat_ann                 = nino34_ann.copy()
dat_ann["nino34_lag1"]  = dat_ann["nino34"].shift(1)
dat_ann["nino3"]        = nino3_ann
dat_ann["nino3_lag1"]   = dat_ann["nino3"].shift(1)
dat_ann["dmi"]          = dmi_ann
dat_ann["dmi_lag1"]     = dat_ann["dmi"].shift(1)


# dat_ann = dat_ann.loc[1950:2023]



#
X = sm.add_constant(dat_ann[["nino3", "nino3_lag1"]])
y = dat_ann["dmi"]
model = sm.OLS(y, X, missing="drop").fit()
dat_ann["dmi_noenso"] = model.resid

dat_ann["dmi_noenso_lag1"] = dat_ann["dmi_noenso"].shift(1)
dat_ann = dat_ann.loc[1950:2023]

# print(dat_ann)
print(np.round(dat_ann.corr(),3))


# save
std = dat_ann["dmi_noenso"].std()
dat_ann["dmi_noenso"] = dat_ann["dmi_noenso"] / std
print(dat_ann["dmi_noenso"])
print(std)
# dat_ann["dmi_noenso"].to_csv("data/dmi_nonino3_ann.csv", header=True) # SAVE



std = dat_ann["dmi"].std()
dat_ann["dmi"] = dat_ann["dmi"] / std
print(std)

#
plt.figure(figsize=(12, 5))
plt.plot(dat_ann.index, dat_ann["nino3"], label="nino3", linewidth=2, alpha=0.7, color='k')
plt.plot(dat_ann.index, dat_ann["dmi"], label="dmi", linewidth=2, alpha=0.7, color='b')
plt.axhline(0,linewidth=1.1, color='k')
plt.xlabel("Year")
plt.ylabel("Index value (s.d.)")
plt.legend()
plt.grid()
plt.tight_layout()
# plt.savefig("/Users/tylerbagwell/Desktop/plot_nino3_vs_dmi.png", dpi=300, bbox_inches="tight")
plt.show()


# x = dat_ann["nino3"]
# y = dat_ann["dmi_noenso_lag0"]
# fig, ax = plt.subplots(figsize=(8,4))
# plot_ccf(x, y, ax=ax, lags=4)
# ax.set_title("Cross-correlations")
# plt.show()

# corr = dat_ann["nino3_lag1"].corr(dat_ann["dmi_noenso_lag0"])
# print(corr)

