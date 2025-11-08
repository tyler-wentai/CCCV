import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
from shapely import wkt
from shapely.geometry import Point
import shapely
from scipy.signal import detrend

print('\n\nSTART ---------------------\n')

dat = pd.read_csv('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_mrsos_mrsosNINO34_Global_square4_19502023.csv')

dat = dat.dropna()
dat = dat.sort_values(['loc_id', 'year']).copy()
g = dat.groupby('loc_id', sort=False)['tp_anom']

for k in range(1, 5):
    dat[f'tp_anom_lag{k}']  = g.shift(k)    # backward lags: t-k
    dat[f'tp_anom_lead{k}'] = g.shift(-k)   # forward lags:  t+k

dat = dat.dropna()
print(dat)
print(dat.shape)

# dat = dat[dat["year"] != 2015]

threshold = 0.1 # remove noise


mask = (dat['conflict_binary'] > 0.0) & (dat['cindex_lag0y'] > +0.5) & (dat['psi_tp_directional'] > +threshold) #el nino
# mask = (dat['conflict_binary'] > 0.0) & (dat['cindex_lag0y'] > -0.5) & (dat['cindex_lag0y'] < +0.5) & (dat['psi_tp_directional'] < -threshold) #neutral
my_onsets = dat.loc[mask]

print("\n...No. of onsets:", my_onsets.shape[0])



import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# columns to include
candidate = (['tp_anom'] +
             [f'tp_anom_lag{k}' for k in range(1, 5)] +
             [f'tp_anom_lead{k}' for k in range(1, 5)])
cols = [c for c in candidate if c in my_onsets.columns]

# offset mapper: lag k -> -k, lead k -> +k, tp_anom -> 0
def tp_offset(name: str):
    if name == 'tp_anom':
        return 0
    m = re.fullmatch(r'tp_anom_(lag|lead)(\d+)', name)
    k = int(m.group(2))
    return -k if m.group(1) == 'lag' else k

# bootstrap CI for mean
def bs_ci_mean(x, B=1000, alpha=0.10, rng=None):
    x = pd.Series(x).dropna().to_numpy()
    n = x.size
    if n == 0:
        return np.nan, np.nan, np.nan
    if rng is None:
        rng = np.random.default_rng(42)
    meds = np.empty(B, float)
    for b in range(B):
        meds[b] = np.mean(x[rng.integers(0, n, n)])
    lo, hi = np.quantile(meds, [alpha/2, 1 - alpha/2])
    return np.mean(x), lo, hi

# compute table
rng = np.random.default_rng(42)
rows = []
for c in cols:
    med, lo, hi = bs_ci_mean(my_onsets[c], B=1000, alpha=0.10, rng=rng)
    rows.append({'col': c, 'offset': tp_offset(c), 'n': my_onsets[c].notna().sum(),
                 'mean': med, 'ci_lo_90': lo, 'ci_hi_90': hi})
res = pd.DataFrame(rows).sort_values('offset').reset_index(drop=True)

print("\n",res)
# sys.exit()

# plot
x = res['offset'].to_numpy()
y = res['mean'].to_numpy()
yerr = np.vstack([y - res['ci_lo_90'].to_numpy(),
                  res['ci_hi_90'].to_numpy() - y])

labels = ['t0' if k == 0 else f't{k:+d}' for k in x]

mask = np.isfinite(y) & np.isfinite(yerr).all(axis=0)
x, y, yerr, labels = x[mask], y[mask], yerr[:, mask], [labels[i] for i in np.where(mask)[0]]

plt.figure()
plt.plot(x,y)
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3)
plt.axhline(0, linewidth=1)
plt.xticks(x, labels)
plt.xlabel('offset')
plt.ylabel('Mean(mrsos_anom)')
plt.title('Mean and 90% CI by lag and lead')
plt.tight_layout()
plt.ylim(-0.8, +0.8)
plt.show()

# res has the numeric results if you need to export
# print(res)

from pathlib import Path
out = Path("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/")
out.mkdir(parents=True, exist_ok=True)

#save
# res.to_csv(out / "SEA_DrierElNinoGridCells_YearlyAnomNeutral_mrsos_mrsosNINO3_Global_square4_19502023.csv", index=False)                     # uncompressed
