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

print('\n\nSTART ---------------------\n')


path = '/Users/tylerbagwell/Desktop/Onset_Binary_GlobalState_DMItype2_mod.f_loyo.csv'
df = pd.read_csv(path)    

df["year_left_out"] = df["year_left_out"] + 1950

print(df.head())

params = ["C0", "C0_2", "C0xPsi", "C0_2xPsi"]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
axes = axes.ravel()

plt.suptitle("IOD & Conflict, State-level, Leave-One-Year-Out\nPosterior means and 90% credible intervals for parameters of climate terms", fontsize=14, horizontalalignment='center')

for ax, p in zip(axes, params):
    x  = df["year_left_out"].to_numpy()
    m  = df[f"mean_{p}"].to_numpy()
    lo = df[f"q05_{p}"].to_numpy()
    hi = df[f"q95_{p}"].to_numpy()

    # plot each year as its own point + interval
    for xi, mi, loi, hii in zip(x, m, lo, hi):
        ax.vlines(xi, loi, hii, color='steelblue', alpha=0.75)
        ax.plot(xi, mi, "o", color='r', ms=3, zorder=3)
    
    if p == "C0":
        ax.set_ylim([-0.5, 0.7])
        ax.set_ylabel(r"$\alpha_1^{(0)}$ (DMI)", fontsize=12)
        ax.axhline(0.09, color='r', linestyle='--', alpha=0.7)
    if p == "C0_2":
        ax.set_ylim([-1.3, 0.7])
        ax.set_ylabel(r"$\alpha_2^{(0)}$ (DMI$^2$)", fontsize=12)
        ax.axhline(-0.34, color='r', linestyle='--', alpha=0.7)
    elif p == "C0xPsi":
        ax.set_ylim([-2.0, 2.0])
        ax.set_ylabel(r"$\beta_1^{(0)}$ (DMI$\times\Psi$)", fontsize=12)
        ax.axhline(0.04, color='r', linestyle='--', alpha=0.7)
    elif p == "C0_2xPsi":
        ax.set_ylim([-0.7, 5.7])
        ax.set_ylabel(r"$\beta_2^{(0)}$ (DMI$^2\times\Psi$)", fontsize=12)
        ax.axhline(2.52, color='r', linestyle='--', alpha=0.7)

    ax.set_xlabel("Year left out")
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color='k', linewidth=1.5, alpha=0.85)

fig.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_loyo_IOD.png', dpi=300, pad_inches=0.01)
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_loyo_IOD.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()