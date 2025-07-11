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

# onsets
onset_path = '/Users/tylerbagwell/Desktop/cccv_data/conflict_datasets/UcdpPrioRice_GeoArmedConflictOnset_v1_CLEANED.csv'
df_onset = pd.read_csv(onset_path)    
gdf_onset = gpd.GeoDataFrame(
    df_onset, 
    geometry=gpd.points_from_xy(df_onset.onset_lon, df_onset.onset_lat),
    crs="EPSG:4326"
)

# teleA
pathA = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_DMItype2_wGeometry.csv'
dfA = pd.read_csv(pathA)
dfA['geometry'] = dfA['geometry'].apply(wkt.loads)
gdfA = gpd.GeoDataFrame(dfA, geometry='geometry')
gdfA.set_crs(epsg=4326, inplace=True)

gdfA_2023 = gdfA[gdfA['year'] == 2023]
gdf_aggA = (
    gdfA_2023
    .groupby('loc_id')
    .agg({
        'geometry': 'first',
        'psi':      'first',
    })
    .reset_index()
)

gdf_aggA = gpd.GeoDataFrame(gdf_aggA, geometry='geometry')
gdf_aggA.set_crs(gdfA.crs, inplace=True)

# teleB
pathB = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Count_Global_DMItype2_square4_wGeometry.csv'
dfB = pd.read_csv(pathB)
dfB['geometry'] = dfB['geometry'].apply(wkt.loads)
gdfB = gpd.GeoDataFrame(dfB, geometry='geometry')
gdfB.set_crs(epsg=4326, inplace=True)

gdf_aggB = gdfB.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first',
}).reset_index()

gdf_aggB = gpd.GeoDataFrame(gdf_aggB, geometry='geometry')
gdf_aggB.set_crs(gdfB.crs, inplace=True)

# teleC
pathC = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_state/Onset_Binary_GlobalState_mrsosDMI_wGeometry.csv'
dfC = pd.read_csv(pathC)
dfC['geometry'] = dfC['geometry'].apply(wkt.loads)
gdfC = gpd.GeoDataFrame(dfC, geometry='geometry')
gdfC.set_crs(epsg=4326, inplace=True)

gdfC_2023 = gdfC[gdfC['year'] == 2023]
gdf_aggC = (
    gdfC_2023
    .groupby('loc_id')
    .agg({
        'geometry': 'first',
        'psi':      'first',
    })
    .reset_index()
)
gdf_aggC = gpd.GeoDataFrame(gdf_aggC, geometry='geometry')
gdf_aggC.set_crs(gdfC.crs, inplace=True)

# teleD
pathD = '/Users/tylerbagwell/Desktop/panel_datasets/onset_datasets_grid/Onset_Binary_Global_mrsosDMI_square4_wGeometry.csv'
dfD = pd.read_csv(pathD)
dfD['geometry'] = dfD['geometry'].apply(wkt.loads)
gdfD = gpd.GeoDataFrame(dfD, geometry='geometry')
gdfD.set_crs(epsg=4326, inplace=True)

gdf_aggD = gdfD.groupby('loc_id').agg({
    'geometry': 'first',
    'psi': 'first',
}).reset_index()

gdf_aggD = gpd.GeoDataFrame(gdf_aggD, geometry='geometry')
gdf_aggD.set_crs(gdfD.crs, inplace=True)



# Define a polygon with lat/lon coordinates
# fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.Robinson()})
# gl = ax.gridlines(
#     crs=ccrs.PlateCarree(),
#     draw_labels={
#         'bottom': True,
#         'left':   True,
#         'top':    False,
#         'right':  False
#     },
#     linewidth=0.4
# )
# gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
# gl.ylocator = mticker.FixedLocator(range(-60, 61, 30))    # parallels every 30°
# gl.xlabel_style = {'size': 10}
# gl.ylabel_style = {'size': 10}
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.top_labels   = False 
# gl.right_labels = False



# # colormap
# bounds = [np.min(gdf_agg['psi']), np.max(gdf_agg['psi'])] #psi_quants
# cmap =  'RdGy_r' #'gist_heat_r'#'PRGn' 

# psi = gdf_agg['psi'].values
# vmin, vmax = psi.min(), psi.max()
# vcenter    = np.median(psi)  # median value for the colormap center

# n_ticks = 15
# boundaries = np.linspace(vmin, vmax, n_ticks)
# labelled_ticks = boundaries[::2]

# norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

# ax.set_global()
# gdf_plot = gdf_agg.plot(
#     column='psi',    
#     cmap=cmap,
#     norm=norm,
#     legend=True,                   
#     legend_kwds={
#         "pad": 0.07,
#         'boundaries': boundaries,
#         'ticks':      labelled_ticks,
#         'orientation': "vertical", 
#         'spacing':    'uniform',        # <— key bit
#         'shrink': 0.65
#     },
#     ax=ax,
#     transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
# )
# ax.add_geometries(gdf_agg['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
# ax.coastlines()

# #
# index_box = mpatches.Rectangle((-150, -5), 60, 10, 
#                         fill=True, facecolor='gray', edgecolor='k', linewidth=1.5, alpha=0.30,
#                         transform=ccrs.PlateCarree())
# ax.add_patch(index_box)

# #
# x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
# ax.scatter(x, y, color='springgreen', edgecolor='black', linewidth=0.75, s=5.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)


# # HISTORGRAM ON COLORBAR
# cbar_ax = gdf_plot.get_figure().axes[-1]
# cbar_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
# cbar_ax.tick_params(labelsize=9)

# data = gdf_agg['psi'].values
# vmin, vmax = cbar_ax.get_ybound()
# bins = np.linspace(vmin, vmax, 15)
# hist, edges = np.histogram(data, bins=bins)
# centers = 0.5 * (edges[:-1] + edges[1:])

# hist_ax = cbar_ax.twiny()

# pad = hist.max() * 0.125
# hist_ax.set_xlim(-pad, hist.max())

# hist_ax.barh(centers, hist,
#              height=(edges[1] - edges[0]),
#              align='center',
#              color='plum',
#              edgecolor='white',
#              linewidth=1.0,
#              alpha=1)

# hist_ax.xaxis.set_ticks_position('bottom')
# hist_ax.xaxis.set_label_position('bottom')
# hist_ax.tick_params(axis='x', labelsize=9)
# hist_ax.set_xlabel('Count', fontsize=9)
# hist_ax.spines['top'].set_visible(False)
# hist_ax.spines['right'].set_visible(False)
# hist_ax.spines['left'].set_visible(False)

# cbar_ax.yaxis.set_ticks_position('left')
# cbar_ax.yaxis.set_label_position('left')
# cbar_ax.set_title("Teleconnection\nstrength", fontsize=9, pad=10)

# ax.text(0.05, 0.98, 'A', transform=ax.transAxes, fontsize=18, bbox=dict(
#         boxstyle='square,pad=0.3',  # try 'square', 'round', 'larrow', etc.
#         facecolor='white',         # box fill color
#         edgecolor='black',         # box edge color
#         linewidth=1                # edge line width
#     ))

# ax.set_title('NINO3 Grid Cell Teleconnection Strength', fontsize=12)
# plt.tight_layout()
# # plt.savefig('/Users/tylerbagwell/Desktop/RobMAP_NINO3type2_psi_raw.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.show()



# ####
# # --- parameters -------------------------------------------------------------
# n_colors   = 256               # total length of the new map (even number is easiest)
# lower_map  = plt.get_cmap('Blues')   # supply any registered colormap here
# upper_map  = plt.get_cmap('Reds')    # ditto
# # ---------------------------------------------------------------------------

# # 1) Build half-length sample grids.
# half = n_colors // 2
# lower_half = lower_map(np.linspace(0.0, 0.5, half, endpoint=False))  # lower half of Blues
# upper_half = upper_map(np.linspace(0.5, 1.0, half))                  # upper half of Reds

# # 2) Concatenate and create a new map.
# combined_colors = np.vstack([lower_half, upper_half])
# blues_reds = mcolors.LinearSegmentedColormap.from_list(
#     name='Blues_Reds',
#     colors=combined_colors
# )

# # 3) (Optional) Register so you can refer to it by name later.
# plt.register_cmap('Blues_Reds', blues_reds)
# ###



def draw_map(ax, var, label, cindex, tele_gdf, spatial_agg_type, cmap, nbin, threshold):
    gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels={
        'bottom': True,
        'left':   True,
        'top':    False,
        'right':  False
    },
    linewidth=0.4)
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 60))  # meridians every 60°
    gl.ylocator = mticker.FixedLocator(range(-60, 61, 30))    # parallels every 30°
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels   = False 
    gl.right_labels = False

    # colormap
    # bounds = [np.min(tele_gdf['psi']), np.max(tele_gdf['psi'])] #psi_quants
    # cmap =  'Reds' #'gist_heat_r'#'PRGn' 

    psi = tele_gdf['psi'].values
    vmin, vmax = psi.min(), psi.max()
    vcenter    = np.median(psi)  # median value for the colormap center

    n_ticks = nbin
    boundaries = np.linspace(vmin, vmax, n_ticks)
    labelled_ticks = boundaries[::2]

    # --- create spliced colormap:
    if var=='teleconnection':
        n_colors   = 256
        lower_map  = plt.get_cmap('PuBu')
        upper_map  = plt.get_cmap('Reds')

        split = (threshold-vmin)/(vmax-vmin)
        
        lhalf = int(n_colors * split) + 1
        uhalf = int(n_colors * (1-split))
        print(lhalf, uhalf)
        lower_half = lower_map(np.linspace(0.0, split, lhalf, endpoint=False))  # lower half of Blues
        upper_half = upper_map(np.linspace(split, 1.0, uhalf))                  # upper half of Reds

        combined_colors = np.vstack([lower_half, upper_half])
        blues_reds = mcolors.LinearSegmentedColormap.from_list(
            name='Blues_Reds',
            colors=combined_colors
        )
        plt.register_cmap('Blues_Reds', blues_reds, override_builtin=True)


    ax.set_global()
    if var=='teleconnection':
        gdf_plot = tele_gdf.plot(
            column='psi',    
            cmap=cmap,
            legend=True,     
            #vmin=-2, vmax=+2,                
            legend_kwds={
                "pad": 0.07,
                'boundaries': boundaries,
                'ticks':      labelled_ticks,
                'orientation': "vertical", 
                'spacing':    'uniform', 
                'shrink': 0.55
            },
            ax=ax,
            transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
        )
    else:
        gdf_plot = tele_gdf.plot(
            column='psi',    
            cmap=cmap,
            legend=True,     
            vmin=-3, vmax=+3,                
            legend_kwds={
                "pad": 0.07,
                'boundaries': boundaries,
                'ticks':      labelled_ticks,
                'orientation': "vertical", 
                'spacing':    'uniform', 
                'shrink': 0.55
            },
            ax=ax,
            transform=ccrs.PlateCarree()  # This tells Cartopy that the data is in lat-lon coordinates
        )
    ax.add_geometries(tele_gdf['geometry'], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='dimgrey', linewidth=0.5)
    ax.coastlines()
    if var!='teleconnection':
        if lab=='c':
            high = gdf_aggA[gdf_aggA['psi'] > threshold]
        elif lab=='d':
            high = gdf_aggB[gdf_aggB['psi'] > threshold]
        ax.add_geometries(high.geometry, crs=ccrs.PlateCarree(), facecolor='none', edgecolor='crimson', linewidth=0.6)

    #
    if cindex == 'NINO3':
        index_box = mpatches.Rectangle((-150, -5), 60, 10, 
                                fill=True, facecolor='purple', edgecolor='purple', linewidth=1.5, alpha=0.20,
                                transform=ccrs.PlateCarree())
        ax.add_patch(index_box)
    elif cindex == 'DMI':
        index_box1 = mpatches.Rectangle(
            (50, -10),  # lower-left corner (longitude, latitude)
            20,         # width: 70E - 50E
            20,         # height: 10N - (-10S)
            fill=True,
            facecolor='purple', edgecolor='purple', 
            linewidth=1.5,
            alpha=0.20,
            transform=ccrs.PlateCarree()
        )
        index_box2 = mpatches.Rectangle(
            (90, -10),  # lower-left corner (longitude, latitude)
            20,         # width: 110E - 90E
            10,         # height: 0 - (-10S)
            fill=True,
            facecolor='purple', edgecolor='purple', 
            linewidth=1.5,
            alpha=0.20,
            transform=ccrs.PlateCarree()
        )
        ax.add_patch(index_box1)
        ax.add_patch(index_box2)
    else:
        raise ValueError("Invalid cindex value. Expected 'NINO3' or 'DMI'.")

    #
    x, y = gdf_onset['onset_lon'].values, gdf_onset['onset_lat'].values
    if var=='teleconnection': dotcolor = 'springgreen'
    else: dotcolor = 'springgreen'
    ax.scatter(x, y, color=dotcolor, edgecolor='black', linewidth=0.75, s=5.0, marker='o', transform=ccrs.PlateCarree(), zorder=5)


    # HISTORGRAM ON COLORBAR
    cbar_ax = gdf_plot.get_figure().axes[-1]
    cbar_ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    cbar_ax.tick_params(labelsize=9)

    data = tele_gdf['psi'].values
    vmin, vmax = cbar_ax.get_ybound()
    bins = np.linspace(vmin, vmax, nbin)
    hist, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    hist_ax = cbar_ax.twiny()

    pad = hist.max() * 0.1
    hist_ax.set_xlim(-pad, hist.max())

    hist_ax.barh(centers, hist,
                height=(edges[1] - edges[0]),
                align='center',
                color='silver',
                edgecolor='white',
                linewidth=1.0,
                alpha=1)
    if var=='teleconnection':
        if   label=='a': yloc1, yloc2 = 0.35, 0.75
        elif label=='b': yloc1, yloc2 = 0.40, 0.80
        hist_ax.text(0.55, yloc1, 'IOD-conflict\nunresponsive\n↓', fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle='square,pad=0.2', linewidth=0, facecolor='gray', alpha=0.0),
                    transform=hist_ax.transAxes, rotation=0)
        hist_ax.text(0.55, yloc2, '↑\nIOD-conflict\nresponsive', fontsize=8, ha="center", va="center", color='k',
                    bbox=dict(boxstyle='square,pad=0.2', linewidth=0, facecolor='gray', alpha=0.0),
                    transform=hist_ax.transAxes, rotation=0)
    else:
        hist_ax.text(0.55, 0.40, 'Dryer in\n+IOD\n↓', fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle='square,pad=0.2', linewidth=0, facecolor='gray', alpha=0.0),
                    transform=hist_ax.transAxes, rotation=0)
        hist_ax.text(0.55, 0.85, '↑\nWetter in\n+IOD', fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle='square,pad=0.2', linewidth=0, facecolor='gray', alpha=0.0),
                    transform=hist_ax.transAxes, rotation=0)
                    

    hist_ax.xaxis.set_ticks_position('bottom')
    hist_ax.xaxis.set_label_position('bottom')
    hist_ax.tick_params(axis='x', labelsize=9)
    hist_ax.set_xlabel('Count', fontsize=9)
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)

    cbar_ax.yaxis.set_ticks_position('left')
    cbar_ax.yaxis.set_label_position('left')
    if var=='teleconnection':
        cbar_ax.set_title("Teleconnection\nstrength", fontsize=9, pad=10)
        hist_ax.axhline(threshold, linewidth=1.0, color='k', linestyle='--')
    else:
        cbar_ax.set_title("Cumulative\ncorrelation", fontsize=9, pad=10)
        hist_ax.axhline(0, linewidth=1.0, color='k', linestyle='--')

    ax.text(0.05, 0.98, label, transform=ax.transAxes, fontsize=14, bbox=dict(
            boxstyle='square,pad=0.2',  # try 'square', 'round', 'larrow', etc.
            facecolor='white',         # box fill color
            edgecolor='black',         # box edge color
            linewidth=0.5                # edge line width
        ))

    if var=='teleconnection':
        title_str = 'DMI Teleconnection, ' + spatial_agg_type
    else:
        title_str = 'Soil Moisture and DMI, ' + spatial_agg_type
    ax.set_title(title_str, fontsize=12)


##
fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 6.5),
    subplot_kw={'projection': ccrs.Robinson()}
)
# fig.subplots_adjust(hspace=-0.5)
axes = axes.flatten()

# Draw each panel with labels A–D
for ax, var, lab, cindex, tele_gdf, spatial_agg_type, cmap, nbin, threshold in zip(axes,
                                     ['teleconnection','teleconnection','mrsos','mrsos'], 
                                     ['a','b','c','d'], 
                                     ['DMI', 'DMI', 'DMI', 'DMI'],
                                     [gdf_aggA, gdf_aggB, gdf_aggC, gdf_aggD],
                                     ['State', 'Grid Cell', 'State', 'Grid Cell'],
                                     ['Blues_Reds', 'Blues_Reds', 'cmo.curl_r', 'cmo.curl_r'], #PuOr
                                     [8,20,8,20],
                                     [0.40, 0.60, 0.40, 0.60]):
    draw_map(ax, var, lab, cindex, tele_gdf, spatial_agg_type, cmap, nbin, threshold)

plt.tight_layout()
plt.savefig('/Users/tylerbagwell/Desktop/manuscript_plots/Main_fig4_v3.png', dpi=300, pad_inches=0.01)
plt.show()