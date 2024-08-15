import json
import numpy as np 
import s3fs
import shapely.wkt as wkt
import xarray as xr
import netCDF4 as nc

from datetime import datetime, timedelta
from collections import OrderedDict as ODict
from database.connection import openDB, closeDB, openCache
from database.elasticache import setData
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3GetTemporaryCredentials, s3Upload, checkReauth
from utils.helpers import get_json, pushBox, getCurationHierarchy
from shapely.geometry import MultiPoint

def getFileList(fs_s3, location, jobInfo, stage, lastDayIsOverlap=False):
    """
    Function to get a list of files from the S3 location
    Get the date range. For curation, we want to pad an extra day on either side.
    :param fs_s3: the open connection to NASA Earthdata S3
    :type fs_s3: S3 object
    :param location: the S3 URI for this dataset
    :type location: str
    :param jobInfo: the dictionary for this data set (either phDef or curation)
    :type jobInfo: dict
    :param stage: either 'phdef' or 'curation'
    :type stage: str
    :param lastDayIsOverlap: tells it if we're overlapping on the last day for chunking
    :type lastDayIsOverlap: bool
    :return files: list of files found on the S3 location
    :rtype files: list
    """
    if stage == "curation":
        endDate   = jobInfo['endDate'] + timedelta(days=1)
        startDate = jobInfo['startDate'] - timedelta(days=1)
    else:
        endDate   = jobInfo['endDate']
        startDate = jobInfo['startDate']
    timeDelta = endDate - startDate
    days = []
    for i in range(timeDelta.days + 1):
        day = startDate + timedelta(days=i)
        days.append(day)

    files = []
    for thisDay in days:
        if 'GPM_3IMERGHH.07' in location:
            fullPath = '%s%s/%s/3B-HHR.MS.MRG.3IMERG.%s*.HDF5' % (location, thisDay.year, thisDay.strftime('%j'), thisDay.strftime('%Y%m%d'))
        elif 'GPM_MERGIR.1' in location:
            fullPath = '%s%s/%s/merg_%s*_4km-pixel.nc4' % (location, thisDay.year, thisDay.strftime('%j'), thisDay.strftime('%Y%m%d'))
        else:
            exit('Bad location.')
        g = fs_s3.glob(fullPath)
        for gf in g:
            files.append(gf)
            if(thisDay == days[-1] and lastDayIsOverlap): # Only want first file of last day in a chunking situation
                break
    
    return files

def rounder(t):
    """
    Function to round to nearest timesteps for this dataset
    :param t: original date and time
    :type t: str
    :return timeIndex: three times (t-1, t, t+1)
    :rtype timeIndex: list
    """
    t = datetime.strptime(t, '%Y%m%d%H%M')
    if t.minute < 15:
        originalT = t.replace(second=0, minute=0)
    elif t.minute >= 15:
        originalT = t.replace(second=0, minute=30)
        
    secs = timedelta(minutes=30).total_seconds()
    plusT = datetime.fromtimestamp(originalT.timestamp() + secs - originalT.timestamp() % secs)
    minusT = datetime.fromtimestamp(originalT.timestamp() - secs - originalT.timestamp() % secs)

    return [minusT.strftime('%Y%m%d%H%M%S'), originalT.strftime('%Y%m%d%H%M%S'), plusT.strftime('%Y%m%d%H%M%S')]


def gpm_reader(jobID):
    """
    Function to read GPM data from NASA's Earthdata Cloud (AWS S3)
    and prepare it for ForTraCC
    :param jobID: job ID to use to submit the request
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, "running")
    jobInfo = getJobInfo(cur, jobID)[0]
    r = openCache()

    # Retrieve credentials and location
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as curDict:
        info = json.load(curDict)
    daac = info[jobInfo['dataset']]['daac']
    location =info[jobInfo['dataset']]['location']
    creds = s3GetTemporaryCredentials(daac) 

    # Set up to access s3
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                               key=creds['accessKeyId'], 
                               secret=creds['secretAccessKey'], 
                               token=creds['sessionToken'])
    
    # Figure out chunking
    isChunk = '-' in str(jobID)
    lastDayIsOverlap = isChunk and (jobInfo['chunkID'] != jobInfo['nChunks']) 
    
    files = getFileList(fs_s3, location, jobInfo, 'phdef', lastDayIsOverlap)


    if len(files) == 0:
        print('No results found. Exiting...')
        updateStatus(db, cur, jobID, "error")
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
        with fs_s3.open(filename, mode='rb') as s3_file_obj:
            print('Reading %s' % filename)
            if jobInfo ['dataset'] == 'GPM_3IMERGHH':
                ds = xr.open_dataset(s3_file_obj, group='Grid')
            elif jobInfo['dataset'] == 'GPM_MERGIR':
                ds = xr.open_dataset(s3_file_obj)
            else:
                exit('Not sure what dataset this is.')
            ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
            if ds['lat'].values.size == 0 or ds['lon'].values.size == 0:
                print('Warning: no valid lats or lons in file %s' % filename)
                continue
            var_data = ds[var]
            if i == 0:
                if jobInfo ['dataset'] == 'GPM_3IMERGHH':
                    startTime = ds['time'].values[0].strftime('%Y-%m-%dT%H:%M:%S.%f')
                elif jobInfo['dataset'] == 'GPM_MERGIR':
                    startTime = ds['time'].values[0].astype(str)[:-3]
                else:
                    exit('Invalid start time.')
                start_time = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%f')
                data['name'] = jobInfo['dataset']
                data['lat'] = np.asarray(ds.lat.values)
                data['lon'] = np.asarray(ds.lon.values)
                data['images'] = ODict()
            times = ds['time']
            for thisTime in times.values:
                thisTimeString = str(thisTime)
                if jobInfo ['dataset'] == 'GPM_3IMERGHH':
                    timeString = datetime.strptime(thisTimeString, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M')
                    data['images'][timeString] = np.asarray(var_data.sel(time=thisTime).transpose())
                elif jobInfo['dataset'] == 'GPM_MERGIR':
                    # time with milliseconds
                    timeString = datetime.strptime(thisTimeString[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d%H%M')
                    data['images'][timeString] = np.asarray(var_data.sel(time=thisTime))
                else:
                    exit('Cannot package images.')
    if 'images' not in data.keys():
        exit('No data in bounds after file reads.')
    if len(data['images']) == 0:
        exit('No data in bounds after file reads.')        

    
    setData(r, data, jobInfo, start_time, jobID) 
    updateStatus(db, cur, jobID, "complete")

    closeDB(db)

    return

def gpm_curator(jobID):
    """
    Function to cruate data for GPM; can do both the GPM_3IMERGHH and GMP_MERIR datasets.
    This will read data from S3, and subset it to the bounds of the anomaly.  It provides
    data for three time steps (t-1, t, t+1) to make sure there is data for temporal interpolation.
    :param jobID: curation jobID
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, "running")
    jobInfo = getJobInfo(cur, jobID)[0]
    phdefJobInfo = getJobInfo(cur, jobInfo['phdefJobID'])[0]
    dataset = jobInfo['dataset']

    with open('/data/code/data-dictionaries/tos2ca-data-collection-dictionary.json') as curDict:
        info = json.load(curDict)
    daac = info[dataset]['daac']
    creds = s3GetTemporaryCredentials(daac)
    location = info[dataset]['location']
    startDate = phdefJobInfo['startDate']
    endDate = phdefJobInfo['endDate']
    timeStep = info[dataset]['timeStep']
    coords = phdefJobInfo['coords']
    variable = jobInfo['variable']
    phdefJobID = jobInfo['phdefJobID']
    units = info[dataset]['units'][variable]
    productInfo = info[dataset]['productInfo']
    fullName = info[dataset]['fullName']

    sql = f'SELECT location, type FROM output WHERE jobID={phdefJobID} AND type IN ("masks", "toc", "hierarchy")'
    print(jobInfo)
    cur.execute(sql)
    results = cur.fetchall()
    for result in results:
        if result['type'] == 'masks':
            maskFile = result['location']
        if result['type'] == 'toc':
            tocFile = result['location']
        if result['type'] == 'hierarchy':
            hierarchyFile = result['location']
    print(startDate)
    print(endDate)
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                                    key=creds['accessKeyId'], 
                                    secret=creds['secretAccessKey'], 
                                    token=creds['sessionToken'])
    files = getFileList(fs_s3, location, phdefJobInfo, 'curation', False)
    print(files)
    closeDB(db)

    #Open the NetCDF-4 file; set global attributes
    ncFilename = '/data/tmp/%s-Curated-Data.nc4' % jobID
    ncFile = nc.Dataset(ncFilename , 'w', format='NETCDF4')
    ncFile.Variable = variable
    ncFile.Dataset = dataset
    ncFile.Units = units
    ncFile.References = 'https://tos2ca-dev1.jpl.nasa.gov'
    ncFile.Project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
    ncFile.Institution = 'NASA Jet Propulsion Laboratory'
    ncFile.ProductionTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ncFile.PhDefJobID = jobInfo['phdefJobID']
    ncFile.ProductInfo = productInfo
    ncFile.FullName = fullName
    ncFile.SpatialCoverage = 'global'
    ncFile.FileFormat = 'NetCDF-4/HDF-5'
    if dataset == 'GPM_MERGIR':
        ncFile.DatasetResolution = '4km x 4km'
    elif dataset == 'GPM_3IMERGHH':
        ncFile.DataResolution = '0.1 x 0.1'
    else: 
        exit('Unknown dataset')

    #Get the grid from the mask file
    fs = s3fs.S3FileSystem()
    ds = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')
    lons = np.asarray(ds['lon'][:])
    lats = np.asarray(ds['lat'][:])
    x, y = np.meshgrid(lons[:], lats[:])
    lon_res = x[0][1] - x[0][0]
    lat_res = y[1][0] - y[0][0]
    ds.close()
    lonShape = lons.shape[0]
    latShape = lats.shape[0]

    #Write to netCDF-4 file
    navGroup = ncFile.createGroup('/navigation')
    latDim = ncFile['navigation'].createDimension('lat', len(lats))
    lonDim = ncFile['navigation'].createDimension('lon', len(lons))
    lat = ncFile['navigation'].createVariable('lat', 'f4', ('lat',), zlib=True, complevel=9)
    lon = ncFile['navigation'].createVariable('lon', 'f4', ('lon',), zlib=True, complevel=9)
    lat[:] = np.asarray(lats, dtype=np.float32)
    lon[:] = np.asarray(lons, dtype=np.float32)
    navGroup.Description = 'The lat and lon are provided here to reconstruct the original global grid from the dataset that created the masks.  They represent the centroid of the grid cell.'

    hierarchyInfo = get_json(hierarchyFile)
    lastMaskGroupName = ''
    curationHierarchy = {}
    for h in hierarchyInfo['masks']:
        #get temp creds for each time so that you don't time out
        newCredsNeeded = checkReauth(creds)
        if newCredsNeeded == 1:
            creds = s3GetTemporaryCredentials(daac)
        #Read the mask file
        print('Using mask time: %s' % h)
        threeTimes = rounder(h)
        print(threeTimes)
        #create mask group for netCDF-4 file using t
        maskGroupName = threeTimes[1]
        if maskGroupName == lastMaskGroupName:
            groupIncrementor += 1
        else:
            groupIncrementor = 1
        inc = str(groupIncrementor)
        curationHierarchy[maskGroupName + '-' + inc] = {}
        print('Creating NetCDF-4 group ' + maskGroupName + '-' + inc)
        maskGroup = ncFile.createGroup(maskGroupName + '-' + inc)
        maskGroup.MaskTime = h
        maskGroup.MaskFileName = 'Anomaly masks from %s' % maskFile
        ds = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + h)
        mask_indices = np.asarray(ds['mask_indices'][:])
        anomalies = []
        for a in ds.mask_indices:
            anomalies.append(a.values[2])
        anomalies = list(set(anomalies))
        anomalies.sort()
        ds.close()
        maskData=np.zeros((latShape, lonShape), dtype=int)
        # loops over the anomalies
        for thisAnomaly in anomalies:
            #create anomaly group for netCDF-4 file using the anomalyID 
            anomalyGroupName = str(thisAnomaly)
            print('Creating NetCDF-4 group ' + anomalyGroupName)
            anomalyGroup = ncFile[maskGroupName + '-' + inc].createGroup(anomalyGroupName)
            curationHierarchy[maskGroupName + '-' + inc][anomalyGroupName] = [variable]
            coords = []
            print("Anomaly # :" + str(thisAnomaly))
            for line in mask_indices:
                i,j,storm_id=line
                lon, lat = lons[j], lats[i]
                lon_idx = int((np.round(float(lon)/lon_res) * lon_res-x[0][0])/lon_res)
                lat_idx = int((np.round(float(lat)/lat_res)* lat_res-y[0][0])/lat_res)
                maskData[lat_idx][lon_idx] = int(storm_id)
                if storm_id == thisAnomaly:
                    coords.append([lon, lat])
            mp = MultiPoint(coords)

            #Choose and read the right GPM file
            variableData = []
            gpmFileList = []
            for thisTime in threeTimes:
                if dataset == 'GPM_3IMERGHH':
                    mTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y%m%d-S%H%M%S-E')
                elif dataset == 'GPM_MERGIR':
                    mTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y%m%d%H')
                else:
                    exit('No good times.')
                print(files)
                print(mTime)
                mFile = list(filter(lambda x: mTime in x, files))[0]
                print(mFile)
                gpmFileList.append(mFile)
                fs_s3 = s3fs.S3FileSystem(anon=False, 
                                    key=creds['accessKeyId'], 
                                    secret=creds['secretAccessKey'], 
                                    token=creds['sessionToken'])
                if dataset == 'GPM_3IMERGHH':
                    with fs_s3.open(mFile, mode='rb') as s3_file_obj:
                        ds = xr.open_dataset(s3_file_obj, group='Grid')
                        min_lon, min_lat, max_lon, max_lat = pushBox(0.2, mp)
                        sliceTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%S')
                        data = ds.sel(time=slice(sliceTime,sliceTime),lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
                        cLat = data.lat
                        cLon = data.lon
                        cVar = data[variable]
                        cTime = data.time
                        variableName = data[variable].name
                        variableAttrs = data[variable].attrs
                        globalAttrs = data.attrs
                        indices = []
                        for t, thisTime in enumerate(cTime):
                            for i, thisLon in enumerate(cLon):
                                for j, thisLat in enumerate(cLat):
                                    indices.append([thisLat, thisLon, data[variable].values[t][i][j]])
                            variableData.append(indices)                        
                elif dataset == 'GPM_MERGIR':
                    with fs_s3.open(mFile, mode='rb') as s3_file_obj:
                        ds = xr.open_dataset(s3_file_obj)
                        min_lon, min_lat, max_lon, max_lat = pushBox(0.2, mp)
                        sliceTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%S')
                        data = ds.sel(time=slice(sliceTime,sliceTime),lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
                        cLat = data.lat
                        cLon = data.lon
                        cVar = data[variable]
                        cTime = data.time
                        variableName = data[variable].name
                        variableAttrs = data[variable].attrs
                        globalAttrs = data.attrs
                        indices = []
                        for t, thisTime in enumerate(cTime):
                            for i, thisLon in enumerate(cLon):
                                for j, thisLat in enumerate(cLat):
                                    indices.append([thisLat, thisLon, data[variable].values[t][j][i]])
                            variableData.append(indices) 
                else:
                    exit('Nothing to read.')
            variableData = np.asarray(variableData)
            print(variableData)
            maskGroup.InputFiles = ','.join(gpmFileList)

            #write anomaly to the netCDF-4
            #not filtering for quality, seems to only be experimental quality filter recommendations
            if len(variableData) != 0:
                observationDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('observation', None)
                dataPointDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('data_point', 3)
                timeStepDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('time_step', 3)
                data = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createVariable(variable, 'f4', ('time_step', 'observation', 'data_point',), zlib=True, complevel=9)
                data[:] = np.asarray(variableData, dtype=np.float32)
                data.Description = 'Each row is (lat,lon,%s); the lat and lon represent the centroid point of the grid cell' % variable
                data.Times = ','.join(threeTimes)
                data.TimeIndexing = 'index 0 = t-1; index 1 = t; index 2 = t+1'
                if dataset == 'GPM_3IMERGHH':
                    data.LongName = variableAttrs['LongName']
                    data.Units = variableAttrs['Units']
                    data.CodeMissingValue = variableAttrs['CodeMissingValue']
                elif dataset == 'GPM_MERGIR':
                    data.LongName = variableAttrs['standard_name']
                    data.Units = variableAttrs['units']
                    data.FillValue = -9999.0

        lastMaskGroupName = maskGroupName

    ncFile.close()

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = ncFilename
    uploadInfo['type'] = 'curated subset'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, 'tos2ca-dev1', db, cur)

    jsonFilename = getCurationHierarchy(jobID, curationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'hierarchy'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, 'tos2ca-dev1', db, cur)
    
    closeDB(db)

    return
