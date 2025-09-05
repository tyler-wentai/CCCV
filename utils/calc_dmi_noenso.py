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
# print(dat_ann["dmi_noenso"])
# print(std)
# dat_ann["dmi_noenso"].to_csv("data/dmi_nonino3_ann.csv", header=True) # SAVE



std = dat_ann["dmi"].std()
dat_ann["dmi"] = dat_ann["dmi"] / std

std = dat_ann["nino3"].std()
dat_ann["nino3"] = dat_ann["nino3"] / std
print(std)

#
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)

# Panel 1
axes[0].plot(dat_ann.index, dat_ann["nino3"], label="NINO3", linewidth=2.5, alpha=0.7, color='k')
axes[0].plot(dat_ann.index, dat_ann["dmi"],   label="DMI",   linewidth=2.5, alpha=0.7, color='red')
axes[0].axhline(0, linewidth=1.1, color='k')
axes[0].set_ylabel("Climate index (s.d.)")
axes[0].grid(True)
axes[0].legend(loc=2)

# Panel 2
axes[1].plot(dat_ann.index, dat_ann["nino3"],       label="NINO3", linewidth=2.5, alpha=0.7, color='k')
axes[1].plot(dat_ann.index, dat_ann["dmi_noenso"],  label="EA-DMI",   linewidth=2.5, alpha=0.7, color='blue')
axes[1].axhline(0, linewidth=1.1, color='k')
axes[1].set_ylabel("Climate index (s.d.)")
axes[1].grid(True)
axes[1].legend()

# Panel 3
axes[2].plot(dat_ann.index, dat_ann["dmi"],         label="DMI", linewidth=2.5, alpha=0.7, color='red')
axes[2].plot(dat_ann.index, dat_ann["dmi_noenso"],  label="EA-DMI", linewidth=2.5, alpha=0.7, color='blue')
axes[2].axhline(0, linewidth=1.1, color='k')
axes[2].set_ylabel("Climate index (s.d.)")
axes[2].set_xlabel("Year")
axes[2].grid(True)
axes[2].legend()

plt.tight_layout(h_pad=3)
plt.savefig("/Users/tylerbagwell/Desktop/SuppFig_allclimateindices.png", dpi=300, bbox_inches="tight")
plt.show()




# x = dat_ann["nino3"]
# y = dat_ann["dmi_noenso_lag0"]
# fig, ax = plt.subplots(figsize=(8,4))
# plot_ccf(x, y, ax=ax, lags=4)
# ax.set_title("Cross-correlations")
# plt.show()

# corr = dat_ann["nino3_lag1"].corr(dat_ann["dmi_noenso_lag0"])
# print(corr)

