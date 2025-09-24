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

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(11, 8), constrained_layout=False)

# 4x4 outer grid
outer = GridSpec(nrows=4, ncols=4, figure=fig, wspace=0.6, hspace=0.6)

# Group 1: rows 0–1, cols 0–1
g1 = outer[0:2, 0:2].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax1  = fig.add_subplot(g1[:, 0])   # spans 2 rows, 1 col
ax2  = fig.add_subplot(g1[0, 1])   # top-right
ax3  = fig.add_subplot(g1[1, 1])   # bottom-right

# Group 2: rows 0–1, cols 2–3
g2 = outer[0:2, 2:4].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax4  = fig.add_subplot(g2[:, 0])
ax5  = fig.add_subplot(g2[0, 1])
ax6  = fig.add_subplot(g2[1, 1])

# Group 3: rows 2–3, cols 0–1
g3 = outer[2:4, 0:2].subgridspec(nrows=2, ncols=2)
ax7  = fig.add_subplot(g3[:,:])

# Group 4: rows 2–3, cols 2–3
g4 = outer[2:4, 2:4].subgridspec(nrows=2, ncols=2, width_ratios=[7, 1], height_ratios=[0.66, 0.33])
ax8 = fig.add_subplot(g4[:, 0])
ax9 = fig.add_subplot(g4[0, 1])
ax10 = fig.add_subplot(g4[1, 1])

# fig.tight_layout()  # optional if you prefer tight spacing
plt.show()


sys.exit()




print('\n\nSTART ---------------------\n')

path_ci1 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Count_Global_NINO3type2_square4.csv"
path_ci2 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/CE_cindex_lag0y_Onset_Count_Global_NINO3type2_square4.csv"
path_ci3 = "/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_Global_DMI_square4.csv"

path1_h = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_GlobalState_NINO3type2_all_ci90_linear.csv'
path2_h = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_GlobalState_mrsosNINO3_wetting_ci90_linear.csv'
path3_h = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_strong_ci90_linear.csv'

path1_l = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_GlobalState_NINO3type2_all_ci90_linear.csv'
path2_l = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/cindex_lag0y_Onset_Binary_GlobalState_mrsosNINO3_drying_ci90_linear.csv'
path3_l = '/Users/tylerbagwell/Documents/Rice_University/CCCV/data/panel_datasets/results_for_onsets/Onset_Binary_GlobalState_DMItype2_weak_ci90_linear.csv'

path_psi = "/Users/tylerbagwell/Desktop/DMItype2_state_pop_avg_psi.csv"
path_psi_eff_pos_weak = "/Users/tylerbagwell/Desktop/margeffect_psi_posiod_0d40.csv"
path_psi_eff_pos = "/Users/tylerbagwell/Desktop/margeffect_psi_posiod_0d80.csv"
path_psi_eff_pos_strong = "/Users/tylerbagwell/Desktop/margeffect_psi_posiod_0d97.csv"

path_psi_eff_neg_weak = "/Users/tylerbagwell/Desktop/margeffect_psi_negiod_0d40.csv"
path_psi_eff_neg_moderate = "/Users/tylerbagwell/Desktop/margeffect_psi_negiod_0d80.csv"
path_psi_eff_neg_strong = "/Users/tylerbagwell/Desktop/margeffect_psi_negiod_0d97.csv"

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

df_c1 = pd.read_csv(path_ci1)
df_c2 = pd.read_csv(path_ci2)
df_c3 = pd.read_csv(path_ci3)

df1_h = pd.read_csv(path1_h)
df2_h = pd.read_csv(path2_h)
df3_h = pd.read_csv(path3_h)

df1_l = pd.read_csv(path1_l)
df2_l = pd.read_csv(path2_l)
df3_l = pd.read_csv(path3_l)

index_closest1_h = df1_h['cindex_lag0y'].abs().idxmin()
index_closest2_h = df2_h['cindex_lag0y'].abs().idxmin()
index_closest3_h = df3_h['cindex_lag0y'].abs().idxmin()
val1_h = df1_h['estimate__'].iloc[index_closest1_h]
val2_h = df2_h['estimate__'].iloc[index_closest2_h]
val3_h = df3_h['estimate__'].iloc[index_closest3_h]

df1_h['estimate__']     = 100*(df1_h['estimate__'])
df1_h['upper__']        = 100*(df1_h['upper__'])
df1_h['lower__']        = 100*(df1_h['lower__'])

df2_h['estimate__']     = 100*(df2_h['estimate__'])
df2_h['upper__']        = 100*(df2_h['upper__'])
df2_h['lower__']        = 100*(df2_h['lower__'])

df3_h['estimate__']     = 100*(df3_h['estimate__'])
df3_h['upper__']        = 100*(df3_h['upper__'])
df3_h['lower__']        = 100*(df3_h['lower__'])

index_closest1_l = df1_l['cindex_lag0y'].abs().idxmin()
index_closest2_l = df2_l['cindex_lag0y'].abs().idxmin()
index_closest3_l = df3_l['cindex_lag0y'].abs().idxmin()
val1_l = df1_l['estimate__'].iloc[index_closest1_l]
val2_l = df2_l['estimate__'].iloc[index_closest2_l]
val3_l = df3_l['estimate__'].iloc[index_closest3_l]

df1_l['estimate__']     = 100*(df1_l['estimate__'])
df1_l['upper__']        = 100*(df1_l['upper__'])
df1_l['lower__']        = 100*(df1_l['lower__'])

df2_l['estimate__']     = 100*(df2_l['estimate__'])
df2_l['upper__']        = 100*(df2_l['upper__'])
df2_l['lower__']        = 100*(df2_l['lower__'])

df3_l['estimate__']     = 100*(df3_l['estimate__'])
df3_l['upper__']        = 100*(df3_l['upper__'])
df3_l['lower__']        = 100*(df3_l['lower__'])


cmap = sns.diverging_palette(220, 20, as_cmap=True)
num_colors = 5
levels = np.linspace(0, 1, num_colors)
colors = [cmap(level) for level in levels]






fig = plt.figure(figsize=(11.0, 8.0), constrained_layout=True)


# Parent: 2 rows
# gs = GridSpec(nrows=2, ncols=1, figure=fig)

# # --- Row 1: add a spacer between col 2 and 3 ---
# gap_top = 0.2  # increase/decrease to taste
# gs_top = gs[0].subgridspec(
#     nrows=1, ncols=5, width_ratios=[7, 1, gap_top, 7, 1]
# )
# ax1 = fig.add_subplot(gs_top[0, 0])
# ax2 = fig.add_subplot(gs_top[0, 1])
# # gs_top[0, 2] is the spacer; no axis created there
# ax3 = fig.add_subplot(gs_top[0, 3])
# ax4 = fig.add_subplot(gs_top[0, 4])

# # --- Row 2: add a spacer between col 1 and 2 ---
# gap_bottom = 0.35
# gs_bottom = gs[1].subgridspec(
#     nrows=1, ncols=4, width_ratios=[8.5, gap_bottom, 7, 1]
# )
# ax5 = fig.add_subplot(gs_bottom[0, 0])
# # gs_bottom[0, 1] is the spacer; no axis created there
# ax6 = fig.add_subplot(gs_bottom[0, 2])
# ax7 = fig.add_subplot(gs_bottom[0, 3])

gs = GridSpec(nrows=4, ncols=1, figure=fig)

# --- Row 1: add a spacer between col 2 and 3 ---
gap_top = 0.2  # increase/decrease to taste
gs_top = gs[0:1].subgridspec(
    nrows=4, ncols=5, width_ratios=[7, 1, gap_top, 7, 1]
)
ax1 = fig.add_subplot(gs_top[0:1, 0])
ax2 = fig.add_subplot(gs_top[0, 1])
# gs_top[0, 2] is the spacer; no axis created there
ax3 = fig.add_subplot(gs_top[0:1, 3])
ax4 = fig.add_subplot(gs_top[0, 4])

# --- Row 2: add a spacer between col 1 and 2 ---
gap_bottom = 0.35
gs_bottom = gs[2:3].subgridspec(
    nrows=1, ncols=4, width_ratios=[8.5, gap_bottom, 7, 1]
)
ax5 = fig.add_subplot(gs_bottom[0:1, 0])
# gs_bottom[0, 1] is the spacer; no axis created there
ax6 = fig.add_subplot(gs_bottom[0:1, 2])
ax7 = fig.add_subplot(gs_bottom[0, 3])


# -- Panel 1 (top) --

ax1h = ax1.twinx()
ax1.set_title('ENSO & Conflict, State', fontsize=12)

# hist + lines
sns.histplot(x=df_c1['cindex_lag0y'], color='gainsboro',
             stat='proportion', bins=12, alpha=1.0, zorder=3,
             ax=ax1)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1_l,
             color='blue', alpha=0.50, ax=ax1h)
ax1h.fill_between(df1_l['cindex_lag0y'],
                 df1_l['lower__'], df1_l['upper__'],
                 color='blue',
                 alpha=0.20, edgecolor=None)
# sns.lineplot(x='cindex_lag0y', y='estimate__', data=df1_h,
#              color='red', ax=ax1h)
# ax1h.fill_between(df1_h['cindex_lag0y'],
#                  df1_h['lower__'], df1_h['upper__'],
#                  color='red',
#                  alpha=0.20, edgecolor=None)
ax1h.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

# swap ticks & labels
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position("right")
ax1.set_yticks([0, 0.1, 0.2])
ax1.set_yticklabels([0, 0.1, 0.2], fontsize=9)
# ax1.set_ylabel("Observed\nproportion", ha='center', fontsize=9)
# ax1.yaxis.set_label_coords(1.07, 0.05)
ax1.set_ylabel("")        # remove the label text
ax1.yaxis.set_visible(True)
ax1.set_xlabel(r"NDJ-averaged NINO3 ($^{\circ}$C)", fontsize=11)
ax1.tick_params(axis='y', direction='in')

ax1h.yaxis.tick_left()
ax1h.yaxis.set_label_position("left")
ax1h.tick_params(axis='y', direction='in')
ax1h.set_ylabel('ACR (p.p.)', fontsize=11)
ax1h.set_yticks([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
# ax1.set_yticklabels([0, 0.1, 0.2], fontsize=9)

# background spans
ax1.axvspan(+1.0, +3.25, color=colors[4], linewidth=0, alpha=0.10, zorder=0)
ax1.axvspan(-1.0, +1.0, color=colors[2], linewidth=0, alpha=0.00, zorder=0)
ax1.axvspan(-2.05, -1.0, color=colors[0], linewidth=0, alpha=0.10, zorder=0)

# annotations
ax1.text(-0.04, 1.02, 'a', transform=ax1.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width

plt.text(0.0, 7.2, 'Neutral',    fontsize=10, ha='center')
plt.text(1.1, 7.2, 'El Niño',       fontsize=10, ha='left')
plt.text(-1.1,7.2, 'La Niña',       fontsize=10, ha='right')

# legend
line_w = mlines.Line2D([], [], color='blue', alpha=0.50)
patch_w = mpatches.Patch(alpha=0.20, color='blue')
# line_s = mlines.Line2D([], [], color='red')
# patch_s = mpatches.Patch(alpha=0.25, color='red')
# handles = [(line_s, patch_s), (line_w, patch_w)]
handles = [(line_w, patch_w)]
labels  = ['All states']
ax1h.legend(handles=handles, labels=labels,
           handler_map={tuple: HandlerTuple(ndivide=1)},
           loc=[0.05,0.65], fontsize=10,
           title=r'', frameon=False,
           title_fontsize=10)

ax1.set_ylim(0, 2.25)
ax1.set_xlim(-2.05, 3.25)
ax1h.set_xlim(-2.05, 3.25)
ax1h.set_ylim(1.5, 8.0)

ax1h.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)



# -- Panel 2 (middle) --
ax3h = ax3.twinx()
ax3.set_title('ENSO & Conflict, State', fontsize=12)

# hist + lines
sns.histplot(x=df_c2['cindex_lag0y'], color='gainsboro',
             stat='proportion', bins=12, alpha=1.0, zorder=3,
             ax=ax3)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_l,
             color='peru', ax=ax3h)
ax3h.fill_between(df2_l['cindex_lag0y'],
                 df2_l['lower__'], df2_l['upper__'],
                 color='peru',
                 alpha=0.45, edgecolor=None)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df2_h,
             color='green', ax=ax3h)
ax3h.fill_between(df2_h['cindex_lag0y'],
                 df2_h['lower__'], df2_h['upper__'],
                 color='green',
                 alpha=0.25, edgecolor=None)
ax3h.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

# swap ticks & labels
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position("right")
ax3.set_yticks([0, 0.1, 0.2])
ax3.set_yticklabels([0, 0.1, 0.2], fontsize=9)
# ax3.set_ylabel("Observed\nproportion", ha='center', fontsize=9)
# ax3.yaxis.set_label_coords(1.07, 0.05)
ax3.set_ylabel("")        # remove the label text
ax3.yaxis.set_visible(True)
ax3.set_xlabel(r"NDJ-averaged NINO3 ($^{\circ}$C)", fontsize=11)
ax3.tick_params(axis='y', direction='in')

ax3h.yaxis.tick_left()
ax3h.yaxis.set_label_position("left")
ax3h.tick_params(axis='y', direction='in')
ax3h.set_ylabel('ACR (p.p.)', fontsize=11)
ax3h.set_yticks([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

# background spans
ax3.axvspan(+1.0, +3.25, color=colors[4], linewidth=0, alpha=0.10, zorder=0)
ax3.axvspan(-1.0, +1.0, color=colors[2], linewidth=0, alpha=0.00, zorder=0)
ax3.axvspan(-2.05, -1.0, color=colors[0], linewidth=0, alpha=0.10, zorder=0)

# annotations
ax3.text(-0.04, 1.02, 'b', transform=ax3.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width

plt.text(0.0, 7.2, 'Neutral',    fontsize=10, ha='center')
plt.text(1.1, 7.2, 'El Niño',       fontsize=10, ha='left')
plt.text(-1.1,7.2, 'La Niña',       fontsize=10, ha='right')

# legend
line_w = mlines.Line2D([], [], color='green')
patch_w = mpatches.Patch(alpha=0.35, color='green')
line_s = mlines.Line2D([], [], color='peru')
patch_s = mpatches.Patch(alpha=0.25, color='peru')
handles = [(line_s, patch_s), (line_w, patch_w)]
labels  = ['Drier in El Niño', 'Wetter in El Niño']
ax3h.legend(handles=handles, labels=labels,
           handler_map={tuple: HandlerTuple(ndivide=1)},
           loc=[0.05,0.65], fontsize=10,
           title=r'', frameon=False,
           title_fontsize=10)

ax3.set_ylim(0, 2.25)
ax3.set_xlim(-2.05, 3.25)
ax3h.set_xlim(-2.05, 3.25)
ax3h.set_ylim(1.5, 8.0)

ax3h.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)


# -- Panel 3 (bottom-left) --

ax5h = ax5.twinx()
ax5.set_title('IOD & Conflict, Dose-response, State', fontsize=12)

# hist + lines
sns.histplot(x=psi['pop_avg_psi'], color='gainsboro',
             stat='count', bins="scott", alpha=1.0, zorder=3,
             ax=ax5)
sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_pos_strong, linestyle='-',
             color='#B91C1C', ax=ax5h, label=r"+$0.97^\circ$C (Max obs.)")
# sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_neg_strong, linestyle='--', alpha=0.3,
#              color='#B91C1C', ax=ax5h)
sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_pos,
             color='#EA580C', ax=ax5h, zorder=4, label=r"$+0.80^\circ$C (2 s.d.)")
# sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_neg_moderate, linestyle='--', alpha=0.3,
#              color='#EA580C', ax=ax5h, zorder=4)
ax5h.fill_between(psi_eff_pos['pop_avg_psi'],
                 psi_eff_pos['conf.low'], psi_eff_pos['conf.high'],
                 color='#EA580C',
                 alpha=0.15, edgecolor=None)
sns.lineplot(x='pop_avg_psi', y='estimate', data=psi_eff_pos_weak, linestyle='-',
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


ax5h.plot([0.43, 0.43], [-0.0258, 0.0227], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Ethiopia
ax5h.plot([0.66, 0.66], [-0.0290, 0.0525], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Somalia
ax5h.plot([0.73, 0.73], [-0.0278, 0.0650], color='grey', linestyle='-', linewidth=0.8, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0230], color='#B91C1C', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0528], color='#B91C1C', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0652], color='#B91C1C', s=10, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0142], color='#EA580C', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0290], color='#EA580C', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0348], color='#EA580C', s=10, zorder=2) # Indonesia

ax5h.scatter([0.43], [0.0035], color='#F59E0B', s=10, zorder=2) # Ethiopia
ax5h.scatter([0.66], [0.0056], color='#F59E0B', s=10, zorder=2) # Somalia
ax5h.scatter([0.73], [0.0063], color='#F59E0B', s=10, zorder=2) # Indonesia

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
ax5.set_yticklabels([0, 25, 50, 75], fontsize=9)
ax5.set_ylabel("No. of states", ha='right', fontsize=9)
ax5.set_xlabel(r"IOD teleconnection strength, $\Psi$", fontsize=11)
ax5.tick_params(axis='y', direction='in')
ax5.yaxis.set_label_coords(1.07, 0.25)

ax5h.yaxis.tick_left()
ax5h.yaxis.set_label_position("left")
ax5h.set_yticks([0, 0.02, 0.04, 0.06])
ax5h.set_yticklabels(["0", "+2", "+4", "+6"], fontsize=11)
ax5h.tick_params(axis='y', direction='in')
ax5h.set_ylabel(r"Change in onset probability,""\n+IOD - neutral (p.p.)", fontsize=11)

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
xs = np.sort(np.asarray(psi['pop_avg_psi'].dropna()))
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
ax5h.set_xlim(xmin, 0.9)
ax5h.xaxis.set_major_locator(FixedLocator(pct_to_x(pct_ticks)))
ax5h.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

# secondary axis
secax = ax5h.secondary_xaxis(-0.2, functions=(x_to_pct, pct_to_x))
secax.xaxis.set_major_locator(FixedLocator(pct_ticks))
secax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))
secax.set_xlabel(r"Percentile rank of IOD teleconnection strength, $\Psi$", fontsize=11)

ax5h.legend(fontsize=10,
           title='+IOD strength', frameon=False,
           title_fontsize=10)


ax5h.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)


# -- Panel 4 (bottom-right) --
ax6h = ax6.twinx()
ax6.set_title('IOD & Conflict, State', fontsize=12)

sns.histplot(x=df_c3['cindex_lag0y'], color='gainsboro',
             stat='proportion', bins=12, alpha=1.0, zorder=3,
             ax=ax6)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df3_l,
             color='dimgray', ax=ax6h)
ax6h.fill_between(df3_l['cindex_lag0y'],
                 df3_l['lower__'], df3_l['upper__'],
                 color='dimgray',
                 alpha=0.30, edgecolor=None)
sns.lineplot(x='cindex_lag0y', y='estimate__', data=df3_h,
             color='red', ax=ax6h)
ax6h.fill_between(df3_h['cindex_lag0y'],
                 df3_h['lower__'], df3_h['upper__'],
                 color='red',
                 alpha=0.20, edgecolor=None)
ax6h.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax6.yaxis.tick_right()
ax6.yaxis.set_label_position("right")
ax6.set_yticks([0, 0.1, 0.2])
ax6.set_yticklabels([0, 0.1, 0.2], fontsize=9)
# ax6.set_ylabel("Observed\nproportion", ha='center', fontsize=9)
# ax6.yaxis.set_label_coords(1.07, 0.05)
ax6.set_ylabel("")        # remove the label text
ax6.yaxis.set_visible(True)
ax6.set_xlabel(r"SON-averaged DMI ($^{\circ}$C)", fontsize=11)
ax6.tick_params(axis='y', direction='in')

ax6h.yaxis.tick_left()
ax6h.yaxis.set_label_position("left")
ax6h.tick_params(axis='y', direction='in')
ax6h.set_ylabel('ACR (p.p.)', fontsize=11)
from matplotlib.ticker import MultipleLocator
ax6h.yaxis.set_major_locator(MultipleLocator(4))

ax6.axvspan(+0.4, +2.5, color=colors[4], linewidth=0, alpha=0.10, zorder=0)
ax6.axvspan(-0.4, +0.4, color=colors[2], linewidth=0, alpha=0.00, zorder=0)
ax6.axvspan(-2.5, -0.4, color=colors[0], linewidth=0, alpha=0.10, zorder=0)

ax6.text(-0.04, 1.02, 'd', transform=ax6.transAxes, ha="center", va="center",
         fontsize=14, bbox=dict(boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',          # box fill color
            edgecolor='black',          # box edge color
            linewidth=0.5))             # edge line width

line_w = mlines.Line2D([], [], color='dimgray')
patch_w = mpatches.Patch(alpha=0.35, color='dimgray')
line_s = mlines.Line2D([], [], color='red')
patch_s = mpatches.Patch(alpha=0.25, color='red')
handles = [(line_s, patch_s), (line_w, patch_w)]
labels  = ['IOD-conflict responsive', 'IOD-conflict unresponsive']
ax6h.legend(handles=handles, labels=labels,
           handler_map={tuple: HandlerTuple(ndivide=1)},
           loc=[0.20,0.65], fontsize=10,
           title=r'', frameon=False,
           title_fontsize=10)

plt.text(0.0, 16.0, 'Neutral',    fontsize=10, ha='center')
plt.text(0.5, 16.0, '+IOD',       fontsize=10, ha='left')
plt.text(-0.5,16.0, '-IOD',       fontsize=10, ha='right')

ax6.set_ylim(0, 2.25)
ax6h.set_ylim(0, 18.0)
ax6h.set_xlim(-1.10, 1.10)

ax6h.grid(
    linestyle='--',   # dashed lines
    linewidth=1.0,     # thinner lines
    alpha=0.1,          # a bit transparent
    color='black',
    zorder=1
)


from matplotlib.ticker import FixedLocator, FormatStrFormatter
#### PLOT AX2

ax2.set_xlim(-1.25, +1.25)
ax2.set_ylim(-1, +1)
ax2.axhline(0.0, linestyle="-", color="k", linewidth=1.0)

ax2.scatter(0.50, 0.326, color='red', s=13)                                 # Teleconnected, NINO3
ax2.plot([0.50, 0.50], [-0.159, 0.824], color='red', linewidth=1.00)        # Teleconnected, NINO3
ax2.scatter(-0.50, 0.349, color='dimgray', s=13)                            # Weakly affected, NINO3
ax2.plot([0-.50, -0.50], [-0.059, 0.763], color='dimgray', linewidth=1.00)  # Weakly affected, NINO3

# ax2.scatter(0.30, 0.429, color='red', s=13)                                                 # Teleconnected, NINO34
# ax2.plot([0.30, 0.30], [-0.027, 0.876], color='red', linestyle="--", linewidth=1.00)        # Teleconnected, NINO34
# ax2.scatter(-0.10, 0.303, color='dimgray', s=13)                                            # Weakly affected, NINO34
# ax2.plot([-0.10, -0.10], [-0.110, 0.730], color='dimgray', linestyle="--", linewidth=1.00)  # Weakly affected, NINO34

ax2.yaxis.set_major_locator(FixedLocator([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

for s in ["top", "right", "bottom"]:
    ax2.spines[s].set_visible(False)

# ax2.set_xlabel(None)
ax2.set_xticks([-0.5, +0.5])
ax2.set_xticklabels(["test1", "test2"], rotation=270)
ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=True)




#### PLOT AX4

ax4.set_xlim(-1.25, +1.25)
ax4.set_ylim(-1, +1)
ax4.axhline(0.0, linestyle="-", color="k", linewidth=1.0)

ax4.scatter(0.50, 0.539, color='peru', s=13)                                                # Drier, NINO3
ax4.plot([0.50, 0.50], [0.188, 0.897], color='peru', linewidth=1.00)                        # Drier, NINO3
ax4.scatter(-0.50, 0.220, color='green', s=13)                                              # Wetter, NINO3
ax4.plot([-0.50, -0.50], [-0.293, 0.735], color='green', linewidth=1.00)                    # Wetter, NINO3

# ax4.scatter(0.30, 0.436, color='peru', s=13)                                                # Drier, NINO34
# ax4.plot([0.30, 0.30], [0.078, 0.791], color='peru', linestyle="--", linewidth=1.00)        # Drier, NINO34
# ax4.scatter(-0.10, 0.347, color='green', s=13)                                              # Wetter, NINO34
# ax4.plot([-0.10, -0.10], [-0.170, 0.873], color='green', linestyle="--", linewidth=1.00)    # Wetter, NINO34

ax4.yaxis.set_major_locator(FixedLocator([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


plt.tight_layout(pad=1.0, w_pad=4.0, h_pad=2.5)
plt.show()


