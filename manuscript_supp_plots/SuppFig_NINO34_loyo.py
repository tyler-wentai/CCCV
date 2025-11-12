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


############# STATE-LEVEL LINEAR
main_path = "/Users/tylerbagwell/Desktop/"
path1 = f'{main_path}Onset_Binary_GlobalState_mrsosNINO34_grouped_linear_wet_loyo.csv'
path2 = f'{main_path}Onset_Binary_GlobalState_mrsosNINO34_grouped_linear_dry_loyo.csv'
path3 = f'{main_path}Onset_Count_Global_mrsosNINO34_square4_grouped_poisson_wet_loyo.csv'
path4 = f'{main_path}Onset_Count_Global_mrsosNINO34_square4_grouped_poisson_dry_loyo.csv'
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
df4 = pd.read_csv(path4)

df1['year_left_out'] += 1950
df2['year_left_out'] += 1950

print(df1.head())

df1['mean_C0']      *= 100
df1['q05_C0']       *= 100
df1['q95_C0']       *= 100
df2['mean_C0']      *= 100
df2['q05_C0']       *= 100
df2['q95_C0']       *= 100

val_a = 0.48
val_b = 0.48
val_c = 0.08
val_d = 0.14


# assume your two DataFrames are df_a and df_b
df_a = df1.sort_values("year_left_out").reset_index(drop=True)
df_b = df2.sort_values("year_left_out").reset_index(drop=True)
df_c = df3.sort_values("year_left_out").reset_index(drop=True)
df_d = df4.sort_values("year_left_out").reset_index(drop=True)

x_a = df_a["year_left_out"].values
x_b = df_b["year_left_out"].values

fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=False)

# A: df_a, C0
y = df_a["mean_C0"].values
yerr = np.vstack([y - df_a["q05_C0"].values, df_a["q95_C0"].values - y])
axs[0, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[0, 0].set_title("Wetter-in-El-Niño group", fontsize=13)
axs[0, 0].set_xlabel("Year removed", fontsize=11)
axs[0, 0].set_ylabel(r"$\beta_1$ (NINO34$_t$)", fontsize=12)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_ylim(-0.2,1.1)
axs[0, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[0, 0].axhline(val_a, color='r', linestyle='--', alpha=0.7)
axs[0, 0].text(1957, -0.1, f"Grouped linear", ha='center', va='top', fontsize=8)

# B: df_b, C0
y = df_b["mean_C0"].values
yerr = np.vstack([y - df_b["q05_C0"].values, df_b["q95_C0"].values - y])
axs[0, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[0, 1].set_title("Drier-in-El-Niño group", fontsize=13)
axs[0, 1].set_xlabel("Year removed", fontsize=11)
axs[0, 1].set_ylabel(r"$\beta_1$ (NINO34$_t$)", fontsize=12)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_ylim(-0.2,1.1)
axs[0, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[0, 1].axhline(val_b, color='r', linestyle='--', alpha=0.7)
axs[0, 1].text(1957, -0.1, f"Grouped linear", ha='center', va='top', fontsize=8)

# C: df_a, C0_2
y = df_c["mean_C0"].values
yerr = np.vstack([y - df_c["q05_C0"].values, df_c["q95_C0"].values - y])
axs[1, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[1, 0].set_xlabel("Year removed", fontsize=11)
axs[1, 0].set_ylabel(r"$\beta_1$ (NINO34$_t$)", fontsize=12)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_ylim(-0.1,0.3)
axs[1, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[1, 0].axhline(val_c, color='r', linestyle='--', alpha=0.7)
axs[1, 0].text(1957, -0.05, f"Grouped Poisson", ha='center', va='top', fontsize=8)

# D: df_b, C0_2
y = df_d["mean_C0"].values
yerr = np.vstack([y - df_d["q05_C0"].values, df_d["q95_C0"].values - y])
axs[1, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[1, 1].set_xlabel("Year removed", fontsize=11)
axs[1, 1].set_ylabel(r"$\beta_1$ (NINO34$_t$)", fontsize=12)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].set_ylim(-0.1,0.3)
axs[1, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[1, 1].axhline(val_d, color='r', linestyle='--', alpha=0.7)
axs[1, 1].text(1957, -0.05, f"Grouped Poisson", ha='center', va='top', fontsize=8)


plt.suptitle("Leave-One-Year-Out\nENSO & Conflict, Grouped Linear & Poisson\nPosterior means and 90% credible intervals for parameters of NINO34 terms", fontsize=14, horizontalalignment='center')

fig.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_NINO34_wetvdry_loyo.png', dpi=300, pad_inches=0.01)
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_NINO34_wetvdry_loyo.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
