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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read csv
df = pd.read_csv("/Users/tylerbagwell/Desktop/avgcomp_from_neutralIOD_DMIlag0y_type2_ensoremoved_90ci_mod.f_withENSO_olddata.csv")

# ensure numeric
for c in ["psi","cindex","estimate","conf.low","conf.high"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["estimate"] *= 100
df[["conf.low","conf.high"]] *= 100

mask = df["cindex"] < 0

# flip estimate
df.loc[mask, "estimate"] *= -1

# flip and swap CI so low <= high
df.loc[mask, ["conf.low","conf.high"]] = -df.loc[mask, ["conf.high","conf.low"]].to_numpy()

# continue as before (pivot, heatmap, contours, etc.)
mat_est = (df.pivot_table(index="psi", columns="cindex",
                      values="estimate", aggfunc="mean")
         .sort_index(axis=0).sort_index(axis=1))
mat_low = (df.pivot_table(index="psi", columns="cindex",
                          values="conf.low", aggfunc="mean")
             .reindex_like(mat_est))
mat_high = (df.pivot_table(index="psi", columns="cindex",
                          values="conf.high", aggfunc="mean")
             .reindex_like(mat_est))

Y = mat_est.index.values        # psi
X = mat_est.columns.values      # cindex
Xg, Yg = np.meshgrid(X, Y)  # cell centers
Z = mat_est.values
Zlow = mat_low.values
Zhigh = mat_high.values
Zsig = np.where(
    ((Zlow >= 0.0) & (Zhigh >= 0.0)) | ((Zlow <= 0.0) & (Zhigh <= 0.0)),
    1.0,
    np.nan
)

# 4) filled contours (+ optional contour lines)
plt.figure(figsize=(8, 4.5))
cf = plt.contourf(Xg, Yg, Z, levels=11, cmap="Reds")   # do not set cmap unless you want a specific palette
cs = plt.contour(Xg, Yg, Z, levels=11, colors='k', linewidths=0.6)
plt.clabel(cs, inline=True, fontsize=10)

# cs0 = plt.contour(Xg, Yg, Zlow, levels=[0.0], linewidths=1.5, linestyles=":", linecolor='blue', alpha=1.0)
# cs0.collections[0].set_label("conf.low = 0")

mask = np.isfinite(Zsig)
plt.scatter(
    Xg[mask], Yg[mask],
    s=10, marker=".", linewidths=0, color="k", alpha=0.6, label="significant (90% CrI)"
)


plt.xticks(fontsize=11)
plt.yticks(fontsize=11)

plt.xlabel(r"IOD magnitude ($^{\circ}$C)", fontsize=12)
plt.ylabel(r"IOD teleconnection strength, $\Psi$", fontsize=12)
plt.title("Predicted change in a state's probability of\nconflict relative to the neutral IOD", fontsize=14)
plt.colorbar(cf, label="Change in probability (p.p.)")


plt.tight_layout()
# plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IODdoseresponse_2D.png', dpi=300, pad_inches=0.01)
# plt.savefig('/Users/tylerbagwell/Desktop/SuppFig_IODdoseresponse_2D.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()
