import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import re
from pathlib import Path

print('\n\nSTART ---------------------\n')

dat = pd.read_csv('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv')
# dat = pd.read_csv('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_mrsos_DMItype2_Global_square4_19502023.csv')

dat = dat.dropna()
dat = dat.sort_values(['loc_id', 'year']).copy()
g = dat.groupby('loc_id', sort=False)['tp_anom']

for k in range(1, 4):
    dat[f'tp_anom_lag{k}']  = g.shift(k)    # backward lags: t-k
dat[f'tp_anom_lead1'] = g.shift(-1)   # forward lag:  t+1


dat = dat.dropna()
# print(dat)
# print(dat.columns)

threshold = 0.5 # remove noise (ENSO-conflict)

# mask = (dat['conflict_binary'] > 0.0) & (dat['psi_tp_directional'] > +threshold) & (dat['cindex_lag0y'] > +0.5) #el nino
# mask = (dat['conflict_binary'] > 0.0) & (dat['psi_tp_directional'] > +threshold) & (dat['cindex_lag0y'] >= -0.5) & (dat['cindex_lag0y'] <= +0.5) #neutral
# mask = (dat['conflict_binary'] > 0.0) & (dat['psi_tp_directional'] < -threshold) & (dat['cindex_lag0y'] < -0.5) #la nina

mask = (dat['conflict_binary'] > 0.0) & (dat['psi_tp_directional'] >= -threshold) & (dat['psi_tp_directional'] <= +threshold) & (dat['cindex_lag0y'] >= -0.5) & (dat['cindex_lag0y'] <= +0.5) #el nino

# mask = (dat['conflict_binary'] > 0.0) & (dat['psi'] > 0.30) & (dat['psi_tp_directional'] < -0.01) & (dat['cindex_lag0y'] > +0.4)
# mask = (dat['conflict_binary'] > 0.0) & (dat['psi'] > 0.30) & (dat['psi_tp_directional'] < -0.01) & (dat['cindex_lag0y'] <= +0.4) & (dat['cindex_lag0y'] >= -0.4)

my_onsets = dat.loc[mask]
print(my_onsets)

print("\n...No. of onsets:", my_onsets.shape[0])

# columns to include
candidate = (['tp_anom'] +
             [f'tp_anom_lag{k}' for k in range(1, 5)] + [f'tp_anom_lead{k}' for k in range(1, 5)])
cols = [c for c in candidate if c in my_onsets.columns]
print(cols)

def tp_offset(name: str, *, allow_only_lead1: bool = True) -> int:
    """
    Returns the x-offset for a column name.
    - tp_anom -> 0
    - tp_anom_lagK -> -K
    - tp_anom_lead1 -> +1
    If allow_only_lead1=True, raises on lead>1 to avoid duplicate x-positions.
    """
    if name == 'tp_anom':
        return 0
    m = re.fullmatch(r'tp_anom_(lag|lead)(\d+)', name)
    if not m:
        raise ValueError(f"Unrecognized name: {name}")
    kind, k = m.group(1), int(m.group(2))
    if kind == 'lag':
        return -k
    # lead
    if allow_only_lead1 and k != 1:
        raise ValueError(f"Only tp_anom_lead1 is allowed, got {name}")
    return +1

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
    med, lo, hi = bs_ci_mean(my_onsets[c], B=5000, alpha=0.05, rng=rng)
    rows.append({'col': c, 'offset': tp_offset(c), 'n': my_onsets[c].notna().sum(),
                 'mean': med, 'ci_lo_90': lo, 'ci_hi_90': hi})
res = pd.DataFrame(rows).sort_values('offset').reset_index(drop=True)

print("\n",res)

### --- plot
x = res['offset'].to_numpy()
y = res['mean'].to_numpy()
yerr = np.vstack([y - res['ci_lo_90'].to_numpy(),
                  res['ci_hi_90'].to_numpy() - y])

labels = ['t0' if k == 0 else f't{k:+d}' for k in x]

mask = np.isfinite(y) & np.isfinite(yerr).all(axis=0)
x, y, yerr, labels = x[mask], y[mask], yerr[:, mask], [labels[i] for i in np.where(mask)[0]]

plt.figure()
plt.plot(x,y, zorder=2)
plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=3, zorder=2)
plt.axhline(0, linewidth=1, zorder=1)
plt.axvline(0, linewidth=1, linestyle='--', zorder=1)
plt.xticks(x, labels)
plt.xlabel('offset')
plt.ylabel('Mean(mrsos_anom)')
plt.title('Mean and 90% CI by lag and lead')
plt.tight_layout()
plt.ylim(-1.2, +1.2)
plt.show()



### --- save
out = Path("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/")
out.mkdir(parents=True, exist_ok=True)
# res.to_csv(out / "SEA_NoeffectElNinoGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv", index=False)                     # uncompressed
