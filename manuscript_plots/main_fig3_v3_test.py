import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerTuple
import cmocean
from matplotlib.gridspec import GridSpec




print('\n\nSTART ---------------------\n')


path_psi = "/Users/tylerbagwell/Desktop/DMItype2_ensoremoved_state_psi.csv"
path_psi_eff_pos_weak = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_posIOD-0d40_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"
path_psi_eff_pos = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_posIOD-0d80_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"
path_psi_eff_pos_strong = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_posIOD-0d97_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"

path_psi_eff_neg_weak = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_posIOD-0d40_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"
path_psi_eff_neg_moderate = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_negIOD-0d80_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"
path_psi_eff_neg_strong = "/Users/tylerbagwell/Desktop/avgcomp_psi_neu_vs_posIOD-0d97_Onset_Binary_GlobalState_DMItype1_ensoremoved.csv"

psi = pd.read_csv(path_psi)
psi_eff_pos_weak = pd.read_csv(path_psi_eff_pos_weak)
psi_eff_pos = pd.read_csv(path_psi_eff_pos)
psi_eff_pos_strong = pd.read_csv(path_psi_eff_pos_strong)

psi = pd.read_csv(path_psi)
psi_eff_neg_weak = pd.read_csv(path_psi_eff_neg_weak)
psi_eff_neg_moderate = pd.read_csv(path_psi_eff_neg_moderate)
psi_eff_neg_strong = pd.read_csv(path_psi_eff_neg_strong)

psi_eff_neg_weak["estimate"]        = - psi_eff_neg_weak["estimate"]
psi_eff_neg_moderate["estimate"]    = - psi_eff_neg_moderate["estimate"]
psi_eff_neg_strong["estimate"]      = - psi_eff_neg_strong["estimate"]



cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]


#######################
#######################

fig = plt.figure(figsize=(11, 7), constrained_layout=False)
gap = -0.05
outer = GridSpec(nrows=5, ncols=4, figure=fig, wspace=0.6, height_ratios=[1, 1, gap, 1, 1],hspace=0.6)

# Group 1: rows 0–1, cols 0–1
g1 = outer[0:2, 0:2].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax1  = fig.add_subplot(g1[:, 0])   # spans 2 rows, 1 col
ax2  = fig.add_subplot(g1[0, 1])   # top-right
axnoA  = fig.add_subplot(g1[1, 1])   # bottom-right

# Group 2: rows 0–1, cols 2–3
g2 = outer[0:2, 2:4].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax3  = fig.add_subplot(g2[:, 0])
ax4  = fig.add_subplot(g2[0, 1])
axnoB  = fig.add_subplot(g2[1, 1])

# Group 3: rows 2–3, cols 0–1
g3 = outer[3:5, 0:2].subgridspec(nrows=2, ncols=2)
ax5  = fig.add_subplot(g3[:,:])

# Group 4: rows 2–3, cols 2–3
g4 = outer[3:5, 2:4].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax6 = fig.add_subplot(g4[:, 0])
ax7 = fig.add_subplot(g4[0, 1])
axnoC = fig.add_subplot(g4[1, 1])


axnoA.set_visible(False)
axnoB.set_visible(False)
axnoC.set_visible(False)



# -- Panel 3 (bottom-left) --

ax5h = ax5.twinx()
ax5.set_title('IOD & Conflict, Dose-response, State', fontsize=11)

# hist + lines
sns.histplot(x=psi['psi'], color='gainsboro',
             stat='count', bins="scott", alpha=1.0, zorder=3,
             ax=ax5)
sns.lineplot(x='psi', y='estimate', data=psi_eff_pos_strong, linestyle='-',
             color='#B91C1C', ax=ax5h, label=r"+$0.97^\circ$C (Max obs.)")
# sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_neg_strong, linestyle='--', alpha=0.3,
#              color='#B91C1C', ax=ax5h)
sns.lineplot(x='psi', y='estimate', data=psi_eff_pos,
             color='#EA580C', ax=ax5h, zorder=4, label=r"$+0.80^\circ$C (2 s.d.)")
sns.lineplot(x='psi', y='estimate', data=psi_eff_neg_moderate, linestyle='--', alpha=0.3,
             color='#EA580C', ax=ax5h, zorder=4)
ax5h.fill_between(psi_eff_pos['psi'],
                 psi_eff_pos['conf.low'], psi_eff_pos['conf.high'],
                 color='#EA580C',
                 alpha=0.15, edgecolor=None)
sns.lineplot(x='psi', y='estimate', data=psi_eff_pos_weak, linestyle='-',
             color='#F59E0B', ax=ax5h, label=r"$+0.40^\circ$C (1 s.d.)")
# sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_neg_weak, linestyle='--', alpha=0.3,
#              color='#F59E0B', ax=ax5h)
# ax5h.fill_between(psi_eff_neg['pop_avg_psi'],
#                  psi_eff_neg['conf.low'], psi_eff_neg['conf.high'],
#                  color='blue',
#                  alpha=0.20, edgecolor=None)
# ax5h.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
import matplotlib.ticker as mtick
ax5h.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{y*100:.0f}"))
ax5h.axhline(0.0, color='k', linestyle="--", zorder=0)


ax5h.plot([0.43, 0.43], [-0.0258, 0.0221], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Ethiopia
ax5h.plot([0.66, 0.66], [-0.0290, 0.0500], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Somalia
ax5h.plot([0.73, 0.73], [-0.0278, 0.0620], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0225], color='#B91C1C', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0505], color='#B91C1C', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0620], color='#B91C1C', s=10, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0140], color='#EA580C', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0279], color='#EA580C', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0332], color='#EA580C', s=10, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0032], color='#F59E0B', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0054], color='#F59E0B', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0062], color='#F59E0B', s=10, zorder=2) # Indonesia

plt.text(0.435, -0.020, "Ethiopia", rotation=270, color='dimgrey', fontsize=7)
plt.text(0.665, -0.020, "Somalia", rotation=270, color='dimgrey', fontsize=7)
plt.text(0.735, -0.021, "Indonesia", rotation=270, color='dimgrey', fontsize=7)

# sns.histplot(x=psi['pop_avg_psi'], color='gainsboro',
#              stat='proportion', bins="scott", alpha=1.0, zorder=5,
#              ax=ax5)

# swap ticks & labels
ax5.yaxis.tick_right()
ax5.yaxis.set_label_position("right")
ax5.set_yticks([0, 25, 50, 75])
ax5.set_yticklabels([0, 25, 50, 75], fontsize=8)
ax5.set_ylabel("No. of states", ha='right', rotation=270, fontsize=8)
ax5.set_xlabel(r"IOD teleconnection strength, $\Psi$", fontsize=10)
ax5.tick_params(axis='y', direction='in')
ax5.yaxis.set_label_coords(1.09, -0.04)

ax5h.yaxis.tick_left()
ax5h.yaxis.set_label_position("left")
ax5h.set_yticks([0, 0.02, 0.04, 0.06])
ax5h.set_yticklabels(["0", "+2", "+4", "+6"], fontsize=11)
ax5h.tick_params(axis='y', direction='in')
ax5h.set_ylabel(r"Change in onset probability,""\n+IOD - neutral IOD (p.p.)", fontsize=10)

# annotations
ax5.text(-0.04, 1.02, 'c', transform=ax5.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width


ax5.set_ylim(0, 400)
ax5h.set_ylim(-0.03, +0.07)
# ax5h.set_xlim(-0.02, 0.92)

from matplotlib.ticker import FixedLocator
xs = np.sort(np.asarray(psi['psi'].dropna()))
xmin, xmax = 0.0, 1.0
xs = xs[(xs >= xmin) & (xs <= xmax)]
N = xs.size
F = (np.arange(1, N + 1) - 0.5) / N

# enforce bijection on [xmin, xmax]
xs_k = np.r_[xmin, xs, xmax]
F_k  = np.r_[0.0,  F,  1.0]

def x_to_pct(x):
    return 100.0 * np.interp(x, xs_k, F_k)

def pct_to_x(pct):
    p = np.asarray(pct) / 100.0
    return np.interp(p, F_k, xs_k)

# primary axis ticks at given percentiles
pct_ticks = [0, 33, 66, 90, 98]#np.arange(0, 101, 20)              # 0,20,...,100
ax5h.set_xlim(xmin, 2.2)
ax5h.xaxis.set_major_locator(FixedLocator(pct_to_x(pct_ticks)))
ax5h.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

# secondary axis
secax = ax5h.secondary_xaxis(-0.2, functions=(x_to_pct, pct_to_x))
secax.xaxis.set_major_locator(FixedLocator(pct_ticks))
secax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
secax.set_xlabel(r"Percentile rank of IOD teleconnection strength, $\Psi$", fontsize=10)

ax5h.legend(fontsize=9,
           title='+IOD magnitude measured by the DMI', frameon=False,
           title_fontsize=9)


ax5h.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)



plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.0)
# plt.savefig('/Users/tylerbagwell/Desktop/Main_fig3_v3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.savefig('/Users/tylerbagwell/Desktop/Main_fig3_v3.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()


