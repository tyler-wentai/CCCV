import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd
from shapely.geometry import Polygon
from shapely import wkt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import xarray as xr
from scipy.stats import norm, gaussian_kde

print('\n\nSTART ---------------------\n')

####################################
####################################

# A
pathA = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_NINO3type2_square4_wGeometry.csv'
dfA = pd.read_csv(pathA)

dfA['geometry'] = dfA['geometry'].apply(wkt.loads)

gdfA = gpd.GeoDataFrame(dfA, geometry='geometry')
gdfA.set_crs(epsg=4326, inplace=True)
gdf_aggA =gdfA.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()

print(gdf_aggA)

thetaA1, muA1, sigmaA1 = 0.628, 0.205, 0.103
thetaA2, muA2, sigmaA2 = 0.372, 0.695, 0.281

psiA_xx = np.linspace(np.min(gdf_aggA['psi']), np.max(gdf_aggA['psi']), 1000)
pdfA1 = thetaA1 * norm.pdf(psiA_xx, loc=muA1, scale=sigmaA1)
pdfA2 = thetaA2 * norm.pdf(psiA_xx, loc=muA2, scale=sigmaA2)

kdeA = gaussian_kde(gdf_aggA['psi'])
nA = gdf_aggA.shape[0]

# B
pathB = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_NINO3type2_wGeometry.csv'
dfB = pd.read_csv(pathB)

dfB['geometry'] = dfB['geometry'].apply(wkt.loads)

gdfB = gpd.GeoDataFrame(dfB, geometry='geometry')
gdfB.set_crs(epsg=4326, inplace=True)
gdf_aggB =gdfB.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first'
}).reset_index()

print(gdf_aggB)

thetaB1, muB1, sigmaB1 = 0.583, 0.231, 0.127
thetaB2, muB2, sigmaB2 = 0.417, 0.785, 0.185

psiB_xx = np.linspace(np.min(gdf_aggB['psi']), np.max(gdf_aggB['psi']), 1000)
pdfB1 = thetaA1 * norm.pdf(psiB_xx, loc=muB1, scale=sigmaB1)
pdfB2 = thetaA2 * norm.pdf(psiB_xx, loc=muB2, scale=sigmaB2)

kdeB = gaussian_kde(gdf_aggB['psi'])
nB = gdf_aggB.shape[0]


# plotting

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)

# === first plot on ax1 ===
ax1.hist(x='psi',
         bins='scott',
         density=True,
         data=gdf_aggA,
         color='silver',
         edgecolor='white',
         linewidth=2.0,
         alpha=1)

ax1.plot(psiA_xx, pdfA1, color='dimgrey', linewidth=2.0,
         label='weakly affected component')
ax1.plot(psiA_xx, pdfA2, color='r', linewidth=2.0,
         label='teleconnected component')
ax1.plot(psiA_xx, kdeA(psiA_xx), color='k', linestyle='--', linewidth=2.0,
         label='KDE of raw data')

ax1.axvline(0.414, color='purple', linewidth=2.0)
ax1.axvspan(0.387, 0.442, color='purple', alpha=0.25, edgecolor='none', linewidth=0)

ax1.set_xlim(0, np.max(gdf_aggA['psi']))
ax1.set_xlabel(r'$\Psi_{NINO3}$', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title(f'Grid cell-level NINO3 teleconnection strengths, N={nA}', fontsize=11)


# === second plot on ax2 (exact same drawing) ===
ax2.hist(x='psi',
         bins='scott',
         density=True,
         data=gdf_aggB,
         color='silver',
         edgecolor='white',
         linewidth=2.0,
         alpha=1)

ax2.plot(psiB_xx, pdfB1, color='dimgrey', linewidth=2.0)
ax2.plot(psiB_xx, pdfB2, color='r', linewidth=2.0)
ax2.plot(psiB_xx, kdeB(psiB_xx), color='k', linestyle='--', linewidth=2.0)

ax2.axvline(0.49, color='purple', linewidth=2.0)
ax2.axvspan(0.407, 0.564, color='purple', alpha=0.25, edgecolor='none', linewidth=0)

ax2.set_xlim(0, np.max(gdf_aggB['psi']))
ax2.set_xlabel(r'$\Psi_{NINO3}$', fontsize=11)
ax2.set_title(f'State-level NINO3 teleconnection strengths, N={nB}', fontsize=11)


# === build combined legend on ax1 only ===
line_proxy = mlines.Line2D([], [], color='purple', linewidth=2.0)
patch_proxy = mpatches.Patch(color='purple', alpha=0.25)

line_proxy1 = mlines.Line2D([], [], color='dimgrey', linewidth=2.0)
line_proxy2 = mlines.Line2D([], [], color='r', linewidth=2.0)
line_proxy3 = mlines.Line2D([], [], color='k', linewidth=2.0, linestyle='--')

handles = [(line_proxy, patch_proxy), (line_proxy1), (line_proxy2), (line_proxy3)]
labels = ['Estimate of 50/50 crossover (with 95% CI)', 
          'Estimate of weakly-affected component',
          'Estimate of teleconnected component',
          'KDE of raw data']
ax1.legend(handles=handles, labels=labels, fontsize=8, frameon=False)
# ax2.legend(handles=handles, labels=labels, fontsize=8, frameon=False)

ax1.text(+0.03, 0.92, 'a', transform=ax1.transAxes, fontsize=14, bbox=dict(
        boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
        facecolor='white',         # box fill color
        edgecolor='black',         # box edge color
        linewidth=0.5                # edge line width
    ))
ax2.text(+0.03, 0.92, 'b', transform=ax2.transAxes, fontsize=14, bbox=dict(
        boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
        facecolor='white',         # box fill color
        edgecolor='black',         # box edge color
        linewidth=0.5                # edge line width
    ))

plt.tight_layout(w_pad=3.0)
plt.savefig('/Users/tylerbagwell/Desktop/manuscript_plots/SuppFig_psiNINO3_normalmixture.png', dpi=300, pad_inches=0.01)
plt.show()

