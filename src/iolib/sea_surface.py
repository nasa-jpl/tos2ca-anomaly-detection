import json
import numpy as np 
import s3fs
import shapely.wkt as wkt
import xarray as xr
import pandas as pd

from datetime import datetime, timedelta
from collections import OrderedDict as ODict
from database.connection import openDB, closeDB, openCache
from database.elasticache import setData
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3GetTemporaryCredentials, checkReauth

def getFileList(jobInfo, creds, location):
    # Compile a list of files
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                        key=creds['accessKeyId'], 
                        secret=creds['secretAccessKey'], 
                        token=creds['sessionToken']) 

    start_of_data = datetime(1992, 10, 10)

    fileList = []
    for date in pd.date_range(jobInfo['startDate'], jobInfo['endDate'], freq='D'): 
        offset = (date - start_of_data).days//5
        date_to_add = (start_of_data + (timedelta(days = 5)*offset))
        filename = f"s3://podaac-ops-cumulus-protected/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL2205/ssh_grids_v2205_{date_to_add.year}{str(date_to_add.month).zfill(2)}{str(date_to_add.day).zfill(2)}12.nc"
        if filename not in fileList:
            fileList.append(filename)

    files = []
    for fullPath in fileList:
        g = fs_s3.glob(fullPath)
        for gf in g:
            files.append(gf)
                    
    return files

def sea_surface_reader(jobID):
    """
    Function to read Sea Surface data from NASA's Earthdata Cloud (AWS S3)
    and prepare it for ForTraCC
    :param jobID: job ID to use to submit the request
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'running')
    jobInfo = getJobInfo(cur, jobID)[0]
    r = openCache()

    # Retrieve the credentials and location
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as phdef:
        info = json.load(phdef)
    daac = info[jobInfo['dataset']]['daac']
    creds = s3GetTemporaryCredentials(daac)

    files = getFileList(jobInfo, creds, info[jobInfo['dataset']]['location'])

    if len(files) == 0:
        print('No results found. Exiting...')
        updateStatus(db, cur, jobID, 'error')
        exit(1)

    # Retrieve the job attributes
    var = jobInfo['variable']
    coords = jobInfo['coords']
    polygon = wkt.loads(coords)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds   

    # PACKAGE THE RESULTS    
    data = {}

    for i, filename in enumerate(files):
        newCredsNeeded = checkReauth(creds)
        if newCredsNeeded == 1:
            creds = s3GetTemporaryCredentials(daac)
        fs_s3 = s3fs.S3FileSystem(anon=False, 
                               key=creds['accessKeyId'], 
                               secret=creds['secretAccessKey'], 
                               token=creds['sessionToken'])    
        with fs_s3.open(filename, mode='rb') as s3_file_obj:
            print('Reading %s' % filename)
            ds = xr.open_dataset(s3_file_obj)
            # Longitude is in 0-360 coordinaters so we need to convert that from the 
            # -180 to 180 that the job polygon bounds are in to get xarray to slice correctly
            ds = ds.sel(Latitude=slice(min_lat,max_lat), Longitude=slice(min_lon+180,max_lon+180))
            if ds['Latitude'].values.size == 0 or ds['Longitude'].values.size == 0:
                print('Warning: no valid lats or lons in file %s' % filename)
                continue
            var_data = ds[var]
            if i == 0:
                startTime = ds['Time'].values[0].astype(str)[:-3]
                start_time = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%f')
                data['name'] = jobInfo['dataset']
                data['lat'] = np.asarray(ds.Latitude.values)
                # Longitude is in 0-360, but we need to convert that to -180 to 180 for ForTraCC
                data['lon'] = np.asarray(ds.Longitude.values-180)
                data['images'] = ODict()
            # Time is a single value for entire value which is # days since start
            times = ds['Time']
            for thisTime in times.values:
                thisTimeString = str(thisTime)
                # time with milliseconds
                timeString = datetime.strptime(thisTimeString[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d%H%M')
                data['images'][timeString] = np.asarray(var_data.sel(Time=thisTime))

    if 'images' not in data.keys():
        exit('No data in bounds after file reads.')
    if len(data['images']) == 0:
        exit('No data in bounds after file reads.')

    closeDB(db)
    setData(r, data, jobInfo, start_time, jobID)

    return