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


def getFileList(phdefJobInfo, creds, location):
    """
    Function to get a list of files from the S3 location
    :param phdefJobInfo: phdef dictionary info for this data set
    :type phdefJobInfo: dict
    :param creds: NASA Earthdata credential information
    :type creds: dict
    :param location: the S3 URI for this dataset
    :type location: str
    :return files: list of files found on the S3 location
    :rtype files: list
    """
    endDate = phdefJobInfo['endDate'] + timedelta(days=1)
    startDate = phdefJobInfo['startDate'] - timedelta(days=1)
    timeDelta = endDate - startDate
    days = []
    for i in range(timeDelta.days + 1):
        day = startDate + timedelta(days=i)
        days.append(day)
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                        key=creds['accessKeyId'], 
                        secret=creds['secretAccessKey'], 
                        token=creds['sessionToken']) 
    files = []


    for thisDay in days:
        fullPath = '%s%s090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc' % (location, thisDay.strftime('%Y%m%d'))
        g = fs_s3.glob(fullPath)
        for gf in g:
            files.append(gf)
                    
    return files


def mur_reader(jobID):
    """
    Function to read MUR data from NASA's Earthdata Cloud (AWS S3)
    and prepare it for ForTraCC
    :param jobID: job ID to use to submit the request
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'running')
    jobInfo = getJobInfo(cur, jobID)[0]
    r = openCache()

    # GET THE CREDENTIALS AND LOCATION
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as phdef:
        info = json.load(phdef)
    daac = info[jobInfo['dataset']]['daac']
    creds = s3GetTemporaryCredentials(daac)
 
    # RETRIEVE THE ATTRIBUTES
    var = jobInfo['variable']
    coords = jobInfo['coords']
    polygon = wkt.loads(coords)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds  

    files = getFileList(jobInfo, creds, info[jobInfo['dataset']]['location'])

    if len(files) == 0:
        print('No results found. Exiting...')
        updateStatus(db, cur, jobID, 'error')
        exit(1)

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
            ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
            if ds['lat'].values.size == 0 or ds['lon'].values.size == 0:
                print('Warning: no valid lats or lons in file %s' % filename)
                continue
            var_data = ds[var]
            if i == 0:
                startTime = ds['time'].values[0].astype(str)[:-3]
                start_time = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%f')
                data['name'] = jobInfo['dataset']
                data['lat'] = np.asarray(ds.lat.values)
                data['lon'] = np.asarray(ds.lon.values)
                data['images'] = ODict()
            # Time is a single value for entire value which is # days since start
            times = ds['time']
            for thisTime in times.values:
                thisTimeString = str(thisTime)
                # time with milliseconds
                timeString = datetime.strptime(thisTimeString[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d%H%M')
                data['images'][timeString] = np.asarray(var_data.sel(time=thisTime))
    
    if 'images' not in data.keys():
        exit('No data in bounds after file reads.')
    if len(data['images']) == 0:
        exit('No data in bounds after file reads.')

    closeDB(db)
    setData(r, data, jobInfo, start_time, jobID)
    
    return

    
    