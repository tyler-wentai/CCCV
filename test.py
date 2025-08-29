# pip install xarray netCDF4 dask cartopy matplotlib
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys

files = [
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1941.1950.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1951.1960.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1961.1970.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1971.1980.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1981.1990.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.1991.2000.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.2001.2010.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.2011.2020.pre.dat.nc",
    "/Users/tylerbagwell/Desktop/cru_ts4.09_pre_data/cru_ts4.09.2021.2024.pre.dat.nc",
]

# Fast path: auto-concat by coords
ds = xr.open_mfdataset(files, combine="by_coords", parallel=True)

print(ds)
sys.exit()

# If you prefer explicit concat:
# dss = [xr.open_dataset(f) for f in files]
# ds = xr.concat(dss, dim="time", data_vars="minimal", coords="minimal", compat="override")

# Ensure unique, sorted time
ds = ds.sortby("time")
if hasattr(ds.indexes["time"], "duplicated"):
    ds = ds.sel(time=~ds.indexes["time"].duplicated())

# Choose variable
var = "pre" if "pre" in ds.data_vars else list(ds.data_vars)[0]
da = ds[var]

# Normalize longitude to [-180, 180)
if "lon" in da.coords and float(da["lon"].max()) > 180:
    lon = ((da["lon"] + 180) % 360) - 180
    da = da.assign_coords(lon=lon).sortby("lon")

# Optional: save combined dataset for reuse
# ds.to_netcdf("cru_ts4.09.2001-2020.pre.nc")

# Plot a given year-month
year, month = 1950, 1
snap = da.sel(time=f"{year}-{month:02d}").squeeze()

proj = ccrs.Robinson()
fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection=proj)
ax.set_global()
ax.coastlines(linewidth=0.5)
img = ax.pcolormesh(
    snap["lon"], snap["lat"], snap,
    transform=ccrs.PlateCarree()
)
cb = plt.colorbar(img, orientation="horizontal", pad=0.05, shrink=0.8)
cb.set_label(f"{var} ({snap.attrs.get('units','')})")
plt.title(f"{var} — {year}-{month:02d}")
plt.tight_layout()
plt.show()
