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
# main_path = "/Users/tylerbagwell/Desktop/"
# path1 = f'{main_path}Onset_Binary_GlobalState_DMItype2_grouped_linear_weaklyaffected_loso.csv'
# path2 = f'{main_path}Onset_Binary_GlobalState_DMItype2_grouped_linear_stronglytele_loso.csv'
# df1 = pd.read_csv(path1)
# df2 = pd.read_csv(path2)

# print(df1.head())

# df1['mean_C0']      *= 100
# df1['mean_C0_2']    *= 100
# df1['q05_C0']       *= 100
# df1['q05_C0_2']     *= 100
# df1['q95_C0']       *= 100
# df1['q95_C0_2']     *= 100
# df2['mean_C0']      *= 100
# df2['mean_C0_2']    *= 100
# df2['q05_C0']       *= 100
# df2['q05_C0_2']     *= 100
# df2['q95_C0']       *= 100
# df2['q95_C0_2']     *= 100

# val_a = 0.912
# val_b = 0.436
# val_c = 0.195
# val_d = 9.020


# # assume your two DataFrames are df_a and df_b
# df_a = df1.sort_values("state_left_out").reset_index(drop=True)
# df_b = df2.sort_values("state_left_out").reset_index(drop=True)

# x_a = df_a["state_left_out"].values
# x_b = df_b["state_left_out"].values

# fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), sharex=False)

# # A: df_a, C0
# y = df_a["mean_C0"].values
# yerr = np.vstack([y - df_a["q05_C0"].values, df_a["q95_C0"].values - y])
# axs[0, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
# axs[0, 0].set_title("Not-strongly-IOD-teleconnected group", fontsize=13)
# axs[0, 0].set_xlabel("State removed", fontsize=11)
# axs[0, 0].tick_params(axis='x', labelrotation=90, labelsize=3)
# axs[0, 0].set_ylabel(r"$\beta_1$ (DMI$_t$)", fontsize=12)
# axs[0, 0].grid(True, alpha=0.3)
# axs[0, 0].set_ylim(-4,4)
# axs[0, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
# axs[0, 0].axhline(val_a, color='r', linestyle='--', alpha=0.7)

# # B: df_b, C0
# y = df_b["mean_C0"].values
# yerr = np.vstack([y - df_b["q05_C0"].values, df_b["q95_C0"].values - y])
# axs[0, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
# axs[0, 1].set_title("Strongly-IOD-teleconnected group", fontsize=13)
# axs[0, 1].set_xlabel("State removed", fontsize=11)
# axs[0, 1].tick_params(axis='x', labelrotation=90, labelsize=5)
# axs[0, 1].set_ylabel(r"$\beta_1$ (DMI$_t$)", fontsize=12)
# axs[0, 1].grid(True, alpha=0.3)
# axs[0, 1].set_ylim(-4,4)
# axs[0, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
# axs[0, 1].axhline(val_b, color='r', linestyle='--', alpha=0.7)

# # C: df_a, C0_2
# y = df_a["mean_C0_2"].values
# yerr = np.vstack([y - df_a["q05_C0_2"].values, df_a["q95_C0_2"].values - y])
# axs[1, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
# axs[1, 0].set_xlabel("State removed", fontsize=11)
# axs[1, 0].tick_params(axis='x', labelrotation=90, labelsize=3)
# axs[1, 0].set_ylabel(r"$\beta_2$ (DMI$_t^2$)", fontsize=12)
# axs[1, 0].grid(True, alpha=0.3)
# axs[1, 0].set_ylim(-4,16)
# axs[1, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
# axs[1, 0].axhline(val_c, color='r', linestyle='--', alpha=0.7)

# # D: df_b, C0_2
# y = df_b["mean_C0_2"].values
# yerr = np.vstack([y - df_b["q05_C0_2"].values, df_b["q95_C0_2"].values - y])
# axs[1, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
# axs[1, 1].set_xlabel("State removed", fontsize=11)
# axs[1, 1].tick_params(axis='x', labelrotation=90, labelsize=5)
# axs[1, 1].set_ylabel(r"$\beta_2$ (DMI$_t^2$)", fontsize=12)
# axs[1, 1].grid(True, alpha=0.3)
# axs[1, 1].set_ylim(-4,16)
# axs[1, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
# axs[1, 1].axhline(val_d, color='r', linestyle='--', alpha=0.7)

# plt.suptitle("Leave-One-State-Out\nIOD & Conflict, State-level, Grouped Linear\nPosterior means and 90% credible intervals for parameters of DMI terms", fontsize=14, horizontalalignment='center')

# fig.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IOD_loso_linear.png', dpi=300, pad_inches=0.01)
# plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IOD_loso_linear.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
# plt.show()





############ GRID CELL-LEVEL POISSON
main_path = "/Users/tylerbagwell/Desktop/"
path1 = f'{main_path}Onset_Binary_GlobalState_DMItype2_grouped_poisson_weaklyaffected_loso.csv'
path2 = f'{main_path}Onset_Binary_GlobalState_DMItype2_grouped_poisson_stronglytele_loso.csv'
df1 = pd.read_csv(path1)
df2 = pd.read_csv(path2)

print(df1.head())

val_a = 0.13
val_b = 0.55
val_c = 0.03
val_d = 1.94


# assume your two DataFrames are df_a and df_b
df_a = df1.sort_values("state_left_out").reset_index(drop=True)
df_b = df2.sort_values("state_left_out").reset_index(drop=True)

x_a = df_a["state_left_out"].values
x_b = df_b["state_left_out"].values

fig, axs = plt.subplots(2, 2, figsize=(13, 6.5), sharex=False)

# A: df_a, C0
y = df_a["mean_C0"].values
yerr = np.vstack([y - df_a["q05_C0"].values, df_a["q95_C0"].values - y])
axs[0, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[0, 0].set_title("Not-strongly-IOD-teleconnected group", fontsize=13)
axs[0, 0].set_xlabel("State removed", fontsize=11)
axs[0, 0].tick_params(axis='x', labelrotation=90, labelsize=3)
axs[0, 0].set_ylabel(r"$\beta_1$ (DMI$_t$)", fontsize=12)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].set_ylim(-0.5,2)
axs[0, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[0, 0].axhline(val_a, color='r', linestyle='--', alpha=0.7)

# B: df_b, C0
y = df_b["mean_C0"].values
yerr = np.vstack([y - df_b["q05_C0"].values, df_b["q95_C0"].values - y])
axs[0, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[0, 1].set_title("Strongly-IOD-teleconnected group", fontsize=13)
axs[0, 1].set_xlabel("State removed", fontsize=11)
axs[0, 1].tick_params(axis='x', labelrotation=90, labelsize=5)
axs[0, 1].set_ylabel(r"$\beta_1$ (DMI$_t$)", fontsize=12)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].set_ylim(-0.5,2)
axs[0, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[0, 1].axhline(val_b, color='r', linestyle='--', alpha=0.7)

# C: df_a, C0_2
y = df_a["mean_C0_2"].values
yerr = np.vstack([y - df_a["q05_C0_2"].values, df_a["q95_C0_2"].values - y])
axs[1, 0].errorbar(x_a, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[1, 0].set_xlabel("State removed", fontsize=11)
axs[1, 0].tick_params(axis='x', labelrotation=90, labelsize=3)
axs[1, 0].set_ylabel(r"$\beta_2$ (DMI$_t^2$)", fontsize=12)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_ylim(-0.8,3.6)
axs[1, 0].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[1, 0].axhline(val_c, color='r', linestyle='--', alpha=0.7)

# D: df_b, C0_2
y = df_b["mean_C0_2"].values
yerr = np.vstack([y - df_b["q05_C0_2"].values, df_b["q95_C0_2"].values - y])
axs[1, 1].errorbar(x_b, y, yerr=yerr, fmt="o", color='r', ecolor='steelblue', markersize=2.5, capsize=0, linewidth=1.2)
axs[1, 1].set_xlabel("State removed", fontsize=11)
axs[1, 1].tick_params(axis='x', labelrotation=90, labelsize=5)
axs[1, 1].set_ylabel(r"$\beta_2$ (DMI$_t^2$)", fontsize=12)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].set_ylim(-0.8,3.6)
axs[1, 1].axhline(0.0, color='k', linewidth=1.5, alpha=0.85)
axs[1, 1].axhline(val_d, color='r', linestyle='--', alpha=0.7)

plt.suptitle("Leave-One-State-Out\nIOD & Conflict, Grid cell-level, Grouped Poisson\nPosterior means and 90% credible intervals for parameters of DMI terms", fontsize=14, horizontalalignment='center')

fig.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IOD_loso_poisson.png', dpi=300, pad_inches=0.01)
plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IOD_loso_poisson.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
