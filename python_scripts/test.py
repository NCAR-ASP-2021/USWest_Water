import intake
import numpy as np
import pandas as pd
import xarray as xr



#this si fucking annoying. you need intake-esm V 2020.11.4 and intake V 0.6.0

#import tensorflow as tf
#import eofs
#from netCDF4 import Dataset
#from eofs.standard import Eof


#import cartopy
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
#import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
#from shapely.geometry.polygon import LinearRing


#import matplotlib as mpl
#import matplotlib.pyplot as plt
#from netCDF4 import Dataset
#from matplotlib import cm

#import copy
#import fsspec
#import pop_tools
import glob


fpath='/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/PSL/'
files = sorted(glob.glob(fpath+'*B1850C5CN*'))

print('...opening dataset...')
bb = xr.open_dataset(files[0])
bb

print(bb)
