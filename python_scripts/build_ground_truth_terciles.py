

# Packages have to go in this order... I don't know why

import numpy as np
import pandas as pd
import xarray as xr
import eofs
from eofs.standard import Eof
import glob



# #this si fucking annoying. you need intake-esm V 2020.11.4 and intake V 0.6.0

# import tensorflow as tf

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from shapely.geometry.polygon import LinearRing


import matplotlib as mpl
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib import cm

import copy
import fsspec
import pop_tools
import intake

from sklearn import linear_model
import sys 
import time

t=int(sys.argv[1])
save_name='/glade/scratch/wchapman/ASP_summerschool/datsets/Daily_Rolling7day_Global_with_GroundTruth_LENS_CTRL_PSL_TS_PRECL_'+str(t)+'_'+str(t+99)+'.nc'
print('...starting on...')
print(save_name)
ds_rolling = xr.open_dataset(save_name)
ds_rolling

#MJO pandas data frame
DF_verif =pd.DataFrame({'year':ds_rolling['time.year'].data,'month':ds_rolling['time.month'].data,'day':ds_rolling['time.day'].data})
DF_verif.head(10)


print('################################')
inits = 0 #reinitialize 1 = yes, 0 = no

PSL_anom = np.array(ds_rolling['PSL_anom']).squeeze()
PRECL_anom = np.array(ds_rolling['PRECL_anom']).squeeze()
TS_anom = np.array(ds_rolling['TS_anom']).squeeze()



if inits==0:
    TS_terc= np.array(ds_rolling['TS_terc']).squeeze()
    PSL_terc = np.array(ds_rolling['PSL_terc']).squeeze()
    PRECL_terc = np.array(ds_rolling['PRECL_terc']).squeeze()
    timy = TS_terc[:,100,100]
    start_loop = np.where(timy==2)[0][-1]

if inits==1:
    TS_terc= np.zeros_like(PSL_anom)
    PSL_terc = np.zeros_like(PSL_anom)
    PRECL_terc = np.zeros_like(PSL_anom)
    start_loop = 0

lat = np.array(ds_rolling['lat']).squeeze()
lon = np.array(ds_rolling['lon']).squeeze()


print('starting at index:',start_loop)
print('....creating indices....')
sys.stdout.flush()
start_time = time.time()

for bb in range(start_loop,len(DF_verif)):
    
    mo = DF_verif['month'][bb]
    da = DF_verif['day'][bb]
    idx = DF_verif[(DF_verif['month']==mo) & (DF_verif['day']==da)].index
    
    if bb % 25==0:
        current_time=time.time()
        elapsed_time = current_time - start_time
        print('time step:',bb,'year:',DF_verif['year'][bb],'day:',da,'month:',mo,'elapsed time:',str(int(elapsed_time))  + " seconds")
        start_time = time.time()
        sys.stdout.flush() 
    ## Temperature 
    TS_anom[idx,:,:]
    TS_day_100 = TS_anom[idx,:,:]
    Period_TS_66=np.percentile(TS_day_100,66.6666,axis=0)
    Period_TS_33=np.percentile(TS_day_100,33.3333,axis=0)
    verif_TS_val= TS_anom[bb,:,:].squeeze()
    #loop to verify. 
    TS_full_verif3366 = np.concatenate([verif_TS_val[None,...],Period_TS_33[None,...],Period_TS_66[None,...]])
    TS_full_verif3366= np.sort(TS_full_verif3366,axis=0)
    
    
    ## PSL 
    PSL_anom[idx,:,:]
    PSL_day_100 = TS_anom[idx,:,:]
    Period_PSL_66=np.percentile(PSL_day_100,66.6666,axis=0)
    Period_PSL_33=np.percentile(PSL_day_100,33.3333,axis=0)
    verif_PSL_val= TS_anom[bb,:,:].squeeze()
    #loop to verify. 
    PSL_full_verif3366 = np.concatenate([verif_PSL_val[None,...],Period_PSL_33[None,...],Period_PSL_66[None,...]])
    PSL_full_verif3366= np.sort(PSL_full_verif3366,axis=0)
    
    
    ## PRECL 
    PRECL_anom[idx,:,:]
    PRECL_day_100 = PRECL_anom[idx,:,:]
    Period_PRECL_66=np.percentile(PRECL_day_100,66.6666,axis=0)
    Period_PRECL_33=np.percentile(PRECL_day_100,33.3333,axis=0)
    verif_PRECL_val= TS_anom[bb,:,:].squeeze()
    #loop to verify. 
    PRECL_full_verif3366 = np.concatenate([verif_PRECL_val[None,...],Period_PRECL_33[None,...],Period_PRECL_66[None,...]])
    PRECL_full_verif3366= np.sort(PRECL_full_verif3366,axis=0)
    
    
    
    
    vertemp = np.zeros([lat.shape[0],lon.shape[0]])
    verPSL = np.zeros([lat.shape[0],lon.shape[0]])
    verPRECL = np.zeros([lat.shape[0],lon.shape[0]])
    for ii in range(lat.shape[0]):
        for jj in range(lon.shape[0]):
            
            if np.isnan(verif_TS_val[ii,jj]):
                vertemp[ii,jj]=np.nan
                verPSL[ii,jj]=np.nan
                verPRECL[ii,jj]=np.nan
            else:
                vertemp[ii,jj] = int(np.where(verif_TS_val[ii,jj]==TS_full_verif3366[:,ii,jj])[0][0])
                verPSL[ii,jj] = int(np.where(verif_PSL_val[ii,jj]==PSL_full_verif3366[:,ii,jj])[0][0])
                verPRECL[ii,jj] = int(np.where(verif_PSL_val[ii,jj]==PRECL_full_verif3366[:,ii,jj])[0][0])
            
            
    TS_terc[bb,:,:] = vertemp
    PSL_terc[bb,:,:] = verPSL
    PRECL_terc[bb,:,:] = verPRECL
    
    if (bb % (365*4)==0) & (bb>0):
        TS_terc=np.expand_dims(TS_terc,axis=0)
        PSL_terc=np.expand_dims(PSL_terc,axis=0)
        PRECL_terc=np.expand_dims(PRECL_terc,axis=0)
        ds_rolling['TS_terc']=(['member_id', 'time', 'lat','lon'],TS_terc)        
        ds_rolling['PSL_terc']=(['member_id', 'time', 'lat','lon'],PSL_terc)
        ds_rolling['PRECL_terc']=(['member_id', 'time', 'lat','lon'],PRECL_terc)


        print('saving in loop:',bb,'year:',DF_verif['year'][bb])
        save_name='/glade/scratch/wchapman/ASP_summerschool/datsets/Daily_Rolling7day_Global_with_GroundTruth_RUN2_LENS_CTRL_PSL_TS_PRECL_'+str(t)+'_'+str(t+99)+'.nc'
        ds_rolling.to_netcdf(save_name)
        print('....saved...')
        sys.stdout.flush()
        TS_terc= TS_terc.squeeze()
        PSL_terc= PSL_terc.squeeze()
        PRECL_terc= PRECL_terc.squeeze()

        
        
        