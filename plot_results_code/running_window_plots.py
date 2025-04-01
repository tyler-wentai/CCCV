import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import sys

path = '/Users/tylerbagwell/Desktop/panel_datasets/results_for_onsets/runningwindow_cindex_lag0y_Onset_Binary_Global_NINO3_square4_high_ratio0.6.csv'
df = pd.read_csv(path)

