source('/glade/scratch/deflorio/all_R_files/mylib.R')

filename = '/glade/scratch/deflorio/IVT_20170213.nc'


#------------------------------
# READ IN SAMPLE IVT NCDF FILE
#------------------------------

library(ncdf4)

a   	= nc_open(filename)
lon  	= a$dim$lon$vals      # longitude (degrees E)
lat	    = a$dim$lat$vals      # latitude  (degrees N)
lead	= a$dim$step$vals     # lead time (hours since initial time)

ivt 	= ncvar_get(a,"ivt")
# can check dimensions of IVT variable using command: dim(ivt).   Dimensions are lon x lat x lead x ensemble member




#------------------------------------------------------------------
# CHOOSE DOMAIN FOR IVT NEAR WESTERN U.S. (20N - 60N, 200E - 250E)
#------------------------------------------------------------------

lon1 = wherenearest(lon,200)
lon2 = wherenearest(lon,250)

lat1 = wherenearest(lat,20)
lat2 = wherenearest(lat,60)


ivt_west = ivt[lon1:lon2,lat1:lat2,,]


#----------------------------------------------------------
# PLOT IVT(x,y) FOR A 7-DAY LEAD TIME AND ENSEMBLE MEMBER
#----------------------------------------------------------


library(maps)


cex.main = 1.3		# title size
cex.axis = 0.9		# axis values size
cex.lab = 1.1		# axis label size

labcex = 1.1		# contour value size
labelcex = 1.1		# colorbar values size

cmap = 'topoREV'	# colormap choice
satmask = 'low'	# saturation choice

zrange = c(0,800)




#pdf('test_ivt_07jul2021.pdf')

mar=c(3,4,5,3)
par(mar=mar)

#note: choose 7th element of "ivt_west" lead time dimension for 7-day lead plot.

ccontour(lon[lon1:lon2],lat[lat1:lat2],ivt_west[,,7,1],zrange=zrange,cmap=cmap,satmask=satmask,xlab='',ylab='',main='ECMWF IVT 7-day forecast for Feb 13 2017 initial forecast')
mcontour(lon[lon1:lon2],lat[lat1:lat2],ivt_west[,,7,1],levels=seq(0,1000,by=100),docolor=F,add=T)
map('world2',add=T)
plot_state_other_lon_branch(add=T)

 mcolorbar(ivt_west,zrange=zrange,cmap=cmap,lon[lon2]+1,lat[lat1],lon[lon2]+2,lat[lat2],satmask=satmask,horiz=F)

#dev.off()