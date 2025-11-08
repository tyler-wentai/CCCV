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

dat = pd.read_csv('/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_mrsos_NINO3type2_Global_square4_19502023.csv')
# dat = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/YearlyAnom_tp_mrsosNINO3_Global_square4_19502023.csv")
# dat.rename(columns={'tp_total': 'tp_anom'}, inplace=True)
print(dat)

dat = dat.dropna()

threshold = 0.01

stddev = np.std(dat['cindex_lag0y'])
print(stddev)

mask = dat['conflict_binary'] > 0.0
mask_no = dat['conflict_binary'] == 0.0

#pos_nina
pos_nina = dat.loc[mask]
mask_help = (pos_nina['psi_tp_directional'] < -threshold) #& (pos_nina['cindex_lag0y'] < -1.0)
pos_nina = pos_nina.loc[mask_help]

#pos_neut
pos_neut = dat.loc[mask]
mask_help = (pos_neut['psi_tp_directional'] > +threshold) & ((pos_neut['cindex_lag0y'] > +1.00))# & (pos_neut['cindex_lag0y'] < 0.00))
pos_neut = pos_neut.loc[mask_help]

#neg_nino
neg_nino = dat.loc[mask]
mask_help = (neg_nino['psi_tp_directional'] < -threshold) & (neg_nino['cindex_lag0y'] > +2.0)
neg_nino = neg_nino.loc[mask_help]

#neg_neut
neg_neut = dat.loc[mask]
mask_help = (neg_neut['psi_tp_directional'] < -threshold) & ((neg_neut['cindex_lag0y'] < -1.00))# & (neg_neut['cindex_lag0y'] > 0.00))
neg_neut = neg_neut.loc[mask_help]



mean1 = np.median(pos_nina['tp_anom'])
mean2 = np.median(neg_nino['tp_anom'])



### --- PLOT 1
cmap = plt.get_cmap('PuOr_r')
num_colors = 9
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]

x1 = pos_nina["tp_anom"]
x11 = pos_neut["tp_anom"]
x2 = neg_nino["tp_anom"]
x22 = neg_neut["tp_anom"]

print(neg_nino[neg_nino['tp_anom']<0].shape)
print(neg_nino[neg_nino['tp_anom']>=0].shape)
print(24/(15+24))

# sys.exit()

combined = np.concatenate([x1, x2])
lo = np.nanmin(combined)
hi = np.nanmax(combined)
span = max(abs(lo), abs(hi))          # symmetric range around 0
bin_edges = np.linspace(-span, span, 8 + 1)  # 8 bins → 9 edges, 0 included

# 2) plot both with the same bin_edges
plt.figure(figsize=(5,4))
plt.hist(x1,
         bins=bin_edges,
         alpha=0.4,
         label="teleconnected & corr>0",
         color=colors[7],
         density=True,
         edgecolor=colors[7])
plt.hist(x2,
         bins=bin_edges,
         alpha=0.4,
         label="teleconnected & corr<0",
         color=colors[1],
         density=True,
         edgecolor=colors[1])
plt.axvline(mean1, color=colors[7], linestyle='--', linewidth=1.5, label=f'{mean1:.3f}')
plt.axvline(mean2, color=colors[1], linestyle=':',  linewidth=1.5, label=f'{mean2:.3f}')

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
proxy1 = (Patch(facecolor=colors[7], edgecolor=colors[7], alpha=0.4),
          Line2D([0], [0], color=colors[7], linestyle='--', linewidth=1.5))
proxy2 = (Patch(facecolor=colors[1], edgecolor=colors[1], alpha=0.4),
          Line2D([0], [0], color=colors[1], linestyle=':',  linewidth=1.5))

plt.legend([proxy1, proxy2],
           [f"corr>0 (M={mean1:.2f})",
            f"corr<0 (M={mean2:.2f})"],
           handler_map={tuple: HandlerTuple(ndivide=None)},
           fontsize=7)


# plt.legend(fontsize=7)
plt.xlabel('← Dryer                   Wetter →\nPrecipitation anomaly in s.d.')
plt.ylabel('Density')
plt.title('Onset-year precipitation anomaly\nENSO all conflicts')
# plt.xlim(-3, +3)

plt.axvline(0., linewidth=1.2, color='k')

# plt.savefig('/Users/tylerbagwell/Desktop/justin_slidedeck/2/NINO3_onsetyear_tpanom_ALL.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.tight_layout()
plt.show()


####
import numpy as np
from scipy.stats import binomtest, wilcoxon

def sign_test_median0(x):
    x = np.asarray(x)
    nz = x[x != 0]                 # ignore ties at 0
    if nz.size == 0:
        return np.nan
    k = np.sum(nz > 0)             # positives
    return binomtest(k, nz.size, 0.5, alternative="two-sided").pvalue

# df has columns 'c1', 'c2'
p1 = sign_test_median0(x1)
p2 = sign_test_median0(x2)

# Wilcoxon one-sample (assumes symmetry about the median)
p1_w = wilcoxon(x1, alternative="less").pvalue
p2_w = wilcoxon(x2, alternative="less").pvalue

# p1_w = wilcoxon(x11, alternative="less").pvalue
# p2_w = wilcoxon(x22, alternative="less").pvalue

print("\n")
print(p1_w)
print(p2_w)

d = x1 - x2
p_sign = sign_test_median0(d)
p_wilc = wilcoxon(d, alternative="two-sided").pvalue

print("\n")
# print(p_sign)
# print(p_wilc)

from scipy.stats import mannwhitneyu
# H0: medians equal (if shapes same). H1: median(x) < median(y)
p_less = mannwhitneyu(x1, x2, alternative="greater", method="auto").pvalue
print(p_less)

import numpy as np
import matplotlib.pyplot as plt

def to1d_no_nan(a):
    arr = np.asarray(getattr(a, "to_numpy", lambda: a)()).ravel()
    return arr[~np.isnan(arr)]

a = to1d_no_nan(x1)  # pos_nina["tp_anom"]
b = to1d_no_nan(x2)  # neg_nino["tp_anom"]
c = to1d_no_nan(x11)
d = to1d_no_nan(x22)

fig, ax = plt.subplots(figsize=(5, 6))
ax.boxplot([a, b, c, d],
           labels=["pos_nina", "neg_nino", "pos_neut", "neg_neut"],
           notch=True,
           showmeans=True)
ax.axhline(0, linewidth=1)  # reference
ax.set_ylabel("tp_anom")
ax.set_title("Box plots: tp_anom by group")
plt.tight_layout()
plt.show()
