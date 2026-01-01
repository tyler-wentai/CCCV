import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
import cartopy.crs as ccrs
from shapely.geometry import Polygon
from shapely import wkt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cmocean
from scipy.stats import pearsonr, spearmanr

print('\n\nSTART ---------------------\n')
pathA1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMItype2_v3_newonsetdata.csv'
dfA1 = pd.read_csv(pathA1)

pathA2 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMItype1_newonsetdata.csv'
dfA2 = pd.read_csv(pathA2)

psiA1 = dfA1.groupby("loc_id")["psi"].first()
psiA2 = dfA2.groupby("loc_id")["psi"].first()

# q66 = np.nanquantile(psiA1, 0.90)          # handles NaNs
# mask = psiA1 > q66
# psiA1 = psiA1[mask]
# psiA2 = psiA2[mask]

xA, yA = psiA1.align(psiA2, join="inner")
mask = xA.notna() & yA.notna()
xA = xA[mask]
yA = yA[mask]

pearson_r, pearson_p = pearsonr(xA.to_numpy(), yA.to_numpy())
spearman_r, spearman_p = spearmanr(xA.to_numpy(), yA.to_numpy())

outA = pd.Series(
    {
        "n_common": len(xA),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
)
print(outA)

#
pathB1 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_v3_square4_newonsetdata.csv'
dfB1 = pd.read_csv(pathB1)
psiB1 = dfB1.groupby("loc_id")["psi"].first()

pathB2 = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype1_square4_newonsetdata.csv'
dfB2 = pd.read_csv(pathB2)
psiB2 = dfB2.groupby("loc_id")["psi"].first()

q66 = np.nanquantile(psiB1, 0.75)          # handles NaNs
print(f"90th percentile of psiB1: {q66}")
# mask = psiB1 > q66
# psiB1 = psiB1[mask]
# psiB2 = psiB2[mask]

xB, yB = psiB1.align(psiB2, join="inner")
mask = xB.notna() & yB.notna()
xB = xB[mask]
yB = yB[mask]

pearson_r, pearson_p = pearsonr(xB.to_numpy(), yB.to_numpy())
spearman_r, spearman_p = spearmanr(xB.to_numpy(), yB.to_numpy())

outB = pd.Series(
    {
        "n_common": len(xB),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
)
print(outB)

import numpy as np

def add_percentile_xaxis(ax, x, offset=36, label="x percentile"):
    x = np.asarray(x)
    xs = np.sort(x[np.isfinite(x)])
    n = xs.size
    if n == 0:
        return None

    # data-units -> percentile (0..100), using empirical CDF
    def x_to_pct(v):
        v = np.asarray(v)
        idx = np.searchsorted(xs, v, side="right")
        return 100.0 * idx / n

    # percentile (0..100) -> data-units, using empirical quantile
    def pct_to_x(p):
        p = np.asarray(p)
        p01 = np.clip(p / 100.0, 0.0, 1.0)
        return np.quantile(xs, p01)

    secax = ax.secondary_xaxis("bottom", functions=(x_to_pct, pct_to_x))
    secax.spines["bottom"].set_position(("outward", offset))
    secax.set_xlabel(label)
    secax.set_xticks([0, 50, 75, 90, 100])
    secax.set_xticklabels(["0%", "50%", "75%", "90%", "100%"])
    
    return secax


def _pearson(x, y):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    r, p = pearsonr(x[m], y[m])
    return r, p, m.sum()

fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))

pairs = [
    (psiA1, psiA2, "State-level"),
    (psiB1, psiB2, "Grid cell-level"),
]

for ax, (x, y, title) in zip(axes, pairs):
    r, p, n = _pearson(x, y)

    ax.scatter(x, y, alpha=0.35, s=17, color="tab:blue", edgecolor="k", linewidth=0.2)
    ax.set_xlabel("Type 1 DMI Teleconnection Strength")
    ax.set_ylabel("Type 2 DMI Teleconnection Strength")
    ax.set_title(title)

    ax.axhline(0, color="gray", linestyle="-", linewidth=1)
    ax.axvline(0, color="gray", linestyle="-", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.7)

    # correlation text inside plot box (top-left)
    ax.set_xlim(min(x)-0.00001, max(x))

    ax.text(
        0.03, 0.97,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="gray")
    )

    add_percentile_xaxis(ax, x, offset=36, label="Percentile Rank of Type 1 DMI Teleconnection Strength")

plt.suptitle("Comparison of Spatial Units' Type 1 vs. Type 2 DMI Teleconnection Strengths", fontsize=12)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Documents/Rice_University/CCCV/SuppFigs/SuppFig_tele_type1_vs_type2_DMI.png', dpi=300, pad_inches=0.01)
plt.show()

