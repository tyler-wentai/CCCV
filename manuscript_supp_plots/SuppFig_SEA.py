import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# df1, df2: columns = ['col','offset','n','mean','ci_lo_90','ci_hi_90']
# If loading from CSVs:
#
df1_a = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_DrierElNinoGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
df2_a = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_WetterElNinoGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
#
df1_b = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_DrierLaNinaGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
df2_b = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_WetterLaNinaGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
#
df1_c = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_DrierNeutralGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
df2_c = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_WetterNeutralGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
#
df1_d = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_NoeffectElNinoGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
df2_d = pd.read_csv("/Users/tylerbagwell/Documents/Rice_University/CCCV/data/cccv_data/misc/SEA_NoeffectLaNinaGridOnsets_threshold0d5_YearlyAnom_mrsos_mrsosNINO3_Global_square4_19502023.csv")
#



def plot_two_ci(df1, df2, label1="Set 1", label2="Set 2",
                phase='neutral', sep=0.125, ax=None):
    # use provided Axes or make one
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.figure

    n1 = int(df1['n'].iloc[0])
    n2 = int(df2['n'].iloc[0])
    label1 = f'{label1} (n={n1})'
    label2 = f'{label2} (n={n2})'

    col1, col2 = 'peru', 'green'

    if phase == 'neutral':
        poly_col, title_help = 'gray', 'during ENSO-neutral'
    elif phase == 'elnino':
        poly_col, title_help = 'red', 'during El Niño'
    elif phase == 'lanina':
        poly_col, title_help = 'royalblue', 'during La Niña'
    elif phase == 'noeffect':
        poly_col, title_help = 'purple', 'for low signed teleconnections group'
        label1 = f'Low signed teleconnections group, El Niño (n={n1})'
        label2 = f'low signed teleconnections group, La Niña (n={n2})'
        col1, col2 = 'lightcoral', 'cornflowerblue'
    else:
        raise ValueError("phase not recognized: use 'neutral' | 'elnino' | 'lanina' | 'noeffect'")

    # sort by offset
    df1 = df1.sort_values("offset").reset_index(drop=True)
    df2 = df2.sort_values("offset").reset_index(drop=True)

    # x positions
    x1 = df1["offset"].to_numpy() - sep/2
    x2 = df2["offset"].to_numpy() + sep/2

    # asymmetric y-error from CIs
    y1 = df1["mean"].to_numpy()
    y2 = df2["mean"].to_numpy()
    yerr1 = np.vstack([y1 - df1["ci_lo_90"].to_numpy(),
                       df1["ci_hi_90"].to_numpy() - y1])
    yerr2 = np.vstack([y2 - df2["ci_lo_90"].to_numpy(),
                       df2["ci_hi_90"].to_numpy() - y2])

    # use ax.* everywhere
    ax.plot(x1, y1, zorder=2, color=col1, linestyle='-', alpha=1.0)
    ax.plot(x2, y2, zorder=2, color=col2, linestyle='-', alpha=0.8)
    ax.errorbar(x1, y1, yerr=yerr1, fmt='o', mfc=col1, mec='k', mew=0.8,
                ecolor=col1, elinewidth=1.2, capsize=3, label=label1)
    ax.errorbar(x2, y2, yerr=yerr2, fmt='s', mfc=col2, mec='k', mew=0.8,
                ecolor=col2, elinewidth=1.2, capsize=3, label=label2)

    # ticks: labels must match ticks
    xticks = np.unique(np.concatenate([df1["offset"].to_numpy(), df2["offset"].to_numpy()])).astype(int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{v:+d}" if v != 0 else "0" for v in xticks])

    ax.set_xlabel("Years from onset", fontsize=9)
    ax.set_ylabel("Mean soil moisture anomaly (s.d.)", fontsize=8)
    ax.axvspan(-0.5, 0.5, facecolor=poly_col, alpha=0.15, zorder=0)
    ax.axhline(0, color='k', linewidth=1.2)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(-0.9, +1.5)
    ax.set_title(f"Conflict onsets {title_help}", fontsize=9)
    ax.text(0., -0.75, f"Onset year", ha='center', va='top', fontsize=6.5)

    # only add per-panel legend if this function created the figure
    ax.legend(loc=[0,0.78], frameon=False, fontsize=7.5)

    return fig, ax


fig, axs = plt.subplots(2, 2, figsize=(7.5, 5), sharex=False, sharey=True, constrained_layout=True)

phases = ["elnino", "lanina", "neutral", "noeffect"]   # replace last with what you need
titles = ["El Niño", "La Niña", "Neutral", "Neutral"]

df1s = [df1_a, df1_b, df1_c, df1_d]
df2s = [df2_a, df2_b, df2_c, df2_d]

for ax, phase, df1s, df2s, ttl in zip(axs.flat, phases, df1s, df2s, titles):
    plot_two_ci(df1s, df2s,
                label1="Drier-in-El-Niño group",
                label2="Wetter-in-El-Niño group",
                phase=phase, sep=0.125, ax=ax)
plt.suptitle('Superposed epoch analysis, ENSO & conflict', fontsize=12)

# shared legend at bottom; leave panel legends off
# handles, labels = axs.flat[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc="lower center", ncol=2)

fig.savefig("/Users/tylerbagwell/Desktop/SEA_NINO3_2x2.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()