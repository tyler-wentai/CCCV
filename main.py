import numpy as np
import sys
import xarray as xr
import matplotlib.pyplot as plt

print('\n\nSTART ---------------------\n')

psi = xr.open_dataarray('/Users/tylerbagwell/Desktop/rho_airVSoni_lag1.nc')
psi['lon'] = xr.where(psi['lon'] > 180, psi['lon'] - 360, psi['lon'])
psi = psi.sortby('lon')

print(psi)

lat = psi['lat']
lon = psi['lon']

print(np.max(psi.values))
print(lon[0])

print(np.unique(psi.values, return_counts=True))

# sys.exit()

from mpl_toolkits.basemap import Basemap, shiftgrid

plt.style.use('ggplot')
fig = plt.figure()

parallels = np.arange(-40,40,20)
meridians = np.arange(-40,80,20)

ax1 = fig.add_subplot(1,1,1)

m = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)

# labels = [left,right,top,bottom]
#m.drawparallels(parallels,labels=[False,True,True,False])
#m.drawmeridians(meridians,labels=[True,False,False,True])
# draw costlines and coutries
m.drawcoastlines(linewidth=0.75)
m.drawcountries(linewidth=0.5)

m.fillcontinents(color='yellow',lake_color='aqua')
# compute the lons and lats to fit the projection
x, y = m(*np.meshgrid(lon,lat))

# draw filled contours.
vmin=np.min(psi.values[0,:,:])  # this sets the colorbar min
vmax=np.max(psi.values[0,:,:]) # this sets the colorbar max
maxval = np.max([np.abs(vmin), np.abs(vmax)])

levels=np.arange(-0.75,+0.75,0.125) # this sets the colorbar levels
ax1 = m.contourf(x,y,psi.values[0,:,:],cmap='coolwarm', vmin=-maxval, vmax=+maxval)
# add colorbar.
cbar = m.colorbar(ax1,location='bottom',pad="10%")
font_size = 8 # Adjust as appropriate.
cbar.ax.tick_params(labelsize=font_size)
cbar.set_label(r'$\rho$ of ONI and Air Temp.') # Python will respond to LaTeX syntax commands.

#plt.title(r'Air Temperature, (K)')
plt.show()