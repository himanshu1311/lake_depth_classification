import pandas as pd
import numpy as np
from datetime import datetime
import ee

#LOAD THE COORDINATE DATA
df = pd.read_csv("/content/drive/MyDrive/glacierMLproject/coord_time_data_final_LT5.csv")

lakeid = np.array(df["lagoslakeid"])
coords = np.array([df["nhd_latitude"], df["nhd_longitude"]])

landsat_bands = np.zeros((coords.shape[0], 9))

#SIGN IN TO GOOGLE EARTH ENGINE
ee.Authenticate()
ee.Initialize()

## FUNCTION TO EXTRACT ALL LANDSAT8 BANDS FOR GIVEN TIME PERIOD AND COORDINATES
def ts_extract(start, lon = None, lat = None,
               end = datetime.today(), stats = 'median'):
   
    # Bands to be extracted for the analysis
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']

    # landsat data url obtained from google earth engine datasets
    landsat_path = "LANDSAT/LC08/C01/T1_RT"
    landsat_ic = ee.ImageCollection(landsat_path)
   
   	#filter the landsat tiles according to date and bands
    landsat = landsat_ic.\
            filterDate(start=start, opt_end=end)\
            .select(bands)
    

    # coordinates of the lake centroid 
    geometry = ee.Geometry.Point(lon, lat)
    x = landsat.filterBounds(geometry).getRegion(geometry, 30).getInfo()
    out = [dict(zip(x[0], values)) for values in x[1:]]
    

    [d.pop('longitude', None) for d in out]
    [d.pop('latitude', None) for d in out]
    [d.pop('time', None) for d in out]

    return out


## ITERATE OVER ALL THE LAKE DATA
for i in range(coords.shape[0]):


  lat = coords[i,0]
  lon = coords[i,1]

  landsat_bands[i,0] = lakeid[i]

  out = ts_extract(lon=lon, lat=lat, sensor='LC8', start=datetime(2013, 4, 4, 0, 1),
                                  end=datetime(2014, 4, 4, 23, 59))
    
    #---------------------------------------Median the landsat values over the entire time duration-------------------------------------

  bandval = np.zeros((len(out),8))
  # bandcount = np.zeros((1,6))
  landsat = np.zeros((1,8))

  for d in range(len(out)):

    banddict = out[d]
    for j in range(6):
      num = banddict["B{0}".format(j+2)]
      if(num != None):
        bandval[d,j] = num

    num1 = banddict["B10"]
    if(num1!= None):
      bandval[d, 6] = num1

    num2 = banddict["B11"]
    if(num2!= None):
      bandval[d, 7] = num2

  landsat = np.median(bandval, axis = 0)
  # for k in range(6):
  #   if(bandcount[0,k] != 0):
  #     landsat[0,k] = bandval[0,k]/bandcount[0,k]

  landsat_bands[i,1:] = landsat

  print(i)


## SAVE THE LANDSAT DATA 
from numpy import savetxt
savetxt('landsat8bands_data.csv', landsat_bands, delimiter=',', fmt='%s')