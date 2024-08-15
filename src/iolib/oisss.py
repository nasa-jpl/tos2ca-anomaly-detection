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
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                        key=creds['accessKeyId'], 
                        secret=creds['secretAccessKey'], 
                        token=creds['sessionToken']) 

    start_of_data = datetime(2011, 8, 24)

    fileList =[]
    for date in pd.date_range(phdefJobInfo['startDate'], phdefJobInfo['endDate'], freq='D'):
        offset = (date - start_of_data).days//4
        date_to_add = start_of_data + timedelta(days=(4 * offset))
        if (date <= date_to_add):
            filename = f"s3://podaac-ops-cumulus-protected/OISSS_L4_multimission_7day_v2/OISSS_L4_multimission_global_7d_v2.0_{date_to_add.year}-{date_to_add.month:02d}-{date_to_add.day:02d}.nc"
            if filename not in fileList:
                fileList.append(filename)

    files = []
    for fullPath in fileList:
        g = fs_s3.glob(fullPath)
        for gf in g:
            files.append(gf)
                    
    return files

def oisss_reader(jobID):
    """
    Function to read OISSS data from NASA's Earthdata Cloud (AWS S3)
    and prepare it for ForTraCC
    :param jobID: job ID to use to submit the request
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'running')
    jobInfo = getJobInfo(cur, jobID)[0]
    r = openCache()

    # Retrieve credentials and location
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

  
    # Package the results
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
            ds = ds.sel(latitude=slice(min_lat,max_lat), longitude=slice(min_lon,max_lon))
            if ds['latitude'].values.size == 0 or ds['longitude'].values.size == 0:
                print('Warning: no valid lats or lons in file %s' % filename)
                continue
            var_data = ds[var]
            if i == 0:
                startTime = ds['time'].values[0].astype(str)[:-3]
                start_time = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%f')
                data['name'] = jobInfo['dataset']
                data['lat'] = np.asarray(ds.latitude.values)
                data['lon'] = np.asarray(ds.longitude.values)
                data['images'] = ODict()
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