import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_ccf
from calc_annual_index import *

print("\n")

start_year = 1950
end_year = 2023


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

print(dat_mon.corr())

x = dat_mon["nino3"]
y = dat_mon["dmi"]
fig, ax = plt.subplots(figsize=(8,4))
plot_ccf(x, y, ax=ax, lags=24)
ax.set_title("Cross-correlations")
plt.show()

# annualized
nino34_ann = compute_annualized_index("nino34", start_year, end_year).set_index("year")
nino34_ann.columns = ["nino34"]

nino3_ann  = compute_annualized_index("nino3",  start_year, end_year).set_index("year")
nino3_ann.columns = ["nino3"]

dmi_ann    = compute_annualized_index("dmi",    start_year, end_year).set_index("year")
dmi_ann.columns = ["dmi"]

dat_ann = nino34_ann.copy()
dat_ann["nino3"] = nino3_ann
dat_ann["dmi"]   = dmi_ann

# print(dat_ann.corr())
x = dat_ann["nino3"]
y = dat_ann["dmi"]
fig, ax = plt.subplots(figsize=(8,4))
plot_ccf(x, y, ax=ax, lags=6)
ax.set_title("Cross-correlations")
plt.show()