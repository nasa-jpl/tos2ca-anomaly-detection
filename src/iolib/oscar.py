import json
import numpy as np 
import s3fs
import xarray as xr
import netCDF4 as nc

from datetime import datetime, timedelta
from database.connection import openDB, closeDB
from utils.helpers import pushBox, getCurationHierarchy, padTimestamps
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3GetTemporaryCredentials, s3Upload, checkReauth
from shapely.geometry import MultiPoint
from utils import tos2ca_secrets
from utils.helpers import convertLons


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
    endDate = phdefJobInfo['endDate'] + timedelta(days=2)
    startDate = phdefJobInfo['startDate'] - timedelta(days=2)
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
        fullPath = '%soscar_currents_final_%s.nc' % (location, thisDay.strftime('%Y%m%d'))
        g = fs_s3.glob(fullPath)
        for gf in g:
            files.append(gf)

    return files

def get_json(filename):
    """
    :param filename: filename of the JSON dictionary
    :type filename: str
    :return j: JSON data from the file
    :rtype j: dict
    """
    fs = s3fs.S3FileSystem()
    with fs.open(filename, 'r') as f:
        j = json.load(f)
    return j


def rounder(t):
    """
    :param t: original date and time
    :type t: str
    :return timeIndex: three times (t-1, t, t+1)
    :rtype timeIndex: list
    """
    t = datetime.strptime(t, '%Y%m%d%H%M')
    originalT = t.replace(second=0, minute=0, hour=0)
        
    plusT = originalT + timedelta(days=1)
    minusT = originalT - timedelta(days=1)

    return [minusT.strftime('%Y%m%d%H%M%S'), originalT.strftime('%Y%m%d%H%M%S'), plusT.strftime('%Y%m%d%H%M%S')]


def oscar_curator(jobID, chunkID):
    """
    Function to cruate data for OSCAR.
    This will read data from S3, and subset it to the bounds of the anomaly.  It provides
    data for three time steps (t-1, t, t+1) to make sure there is data for temporal interpolation.
    :param jobID: curation jobID 
    :type jobID: int
    :param chunkID: curation chunkID
    :type chunkID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'running')
    updateStatus(db, cur, jobID, 'subsetting', chunkID=chunkID, jobStart=True)
    jobInfo = getJobInfo(cur, jobID, chunkID)[0]
    phdefJobInfo = getJobInfo(cur, jobInfo['phdefJobID'])[0]
    dataset = jobInfo['dataset']
    nChunks = jobInfo['nChunks']

    with open('/data/code/data-dictionaries/tos2ca-data-collection-dictionary.json') as curDict:
        info = json.load(curDict)
    daac = info[dataset]['daac']
    creds = s3GetTemporaryCredentials(daac)
    location = info[dataset]['location']
    startDate = jobInfo['startDate']
    endDate = jobInfo['endDate']
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
    files = getFileList(phdefJobInfo, creds, location)
    print(files)
    closeDB(db)

    #Open the NetCDF-4 file; set global attributes
    ncFilename = '/data/tmp/%s-%s-Curated-Data.nc4' % (jobID, chunkID)
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
    ncFile.DataResolution = '0.25 x 0.25'

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
    hTimes = hierarchyInfo['masks']
    hierarchyTimes = {}
    for thisHTime in hTimes.keys():
        ht = datetime.strptime(thisHTime, '%Y%m%d%H%M')
        if ht >= startDate and ht <= endDate:
            hierarchyTimes[thisHTime] = ['mask_indices']
    if chunkID == 1 and nChunks > 1:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'days','quantity':1}, first=True)
    elif chunkID == nChunks and nChunks > 1:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'days','quantity':1}, last=True)
    elif nChunks == 1:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'days','quantity':1}, first=True, last=True)
    else:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'days','quantity':1})
    print(newTimestamps)
    for h in newTimestamps:
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
        readH = h
        if chunkID == 1 and h == list(newTimestamps.keys())[0]:
            readH = list(newTimestamps.keys())[1]
        elif chunkID == nChunks and h == list(newTimestamps.keys())[-1]:
            readH = list(newTimestamps.keys())[-2]
        else:
            readH = h
        ds = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + readH)
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
            #TODO: make this a function where it gets the polygon
            for line in mask_indices:
                i,j,storm_id=line
                lon, lat = lons[j], lats[i]
                lon_idx = int((np.round(float(lon)/lon_res) * lon_res-x[0][0])/lon_res)
                lat_idx = int((np.round(float(lat)/lat_res)* lat_res-y[0][0])/lat_res)
                maskData[lat_idx][lon_idx] = int(storm_id)
                if storm_id == thisAnomaly:
                    coords.append([lon, lat])
            mp = MultiPoint(coords)
            
            #Choose and read the right OSCAR file
            variableData = []
            oscarFileList = []
            for thisTime in threeTimes:
                mTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y%m%d')
                print(mTime)
                print(files)
                print(mTime)
                mFile = list(filter(lambda x: mTime in x, files))[0]
                print(mFile)
                oscarFileList.append(mFile)
                # check credentials so you don't time out
                newCredsNeeded = checkReauth(creds)
                if newCredsNeeded == 1:
                    creds = s3GetTemporaryCredentials(daac)
                fs_s3 = s3fs.S3FileSystem(anon=False, 
                                    key=creds['accessKeyId'], 
                                    secret=creds['secretAccessKey'], 
                                    token=creds['sessionToken'])
                with fs_s3.open(mFile, mode='rb') as s3_file_obj:
                    ds = xr.open_dataset(s3_file_obj)
                    min_lon, min_lat, max_lon, max_lat = pushBox(0.5, mp)
                    sliceTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
                    #OSCAR longitudes are 0-360
                    min_fix_lon, max_fix_lon = convertLons(min_lon, max_lon)
                    min_lat = round(min_lat * 4) / 4
                    max_lat = round(max_lat * 4) / 4
                    min_fix_lon = round(min_fix_lon * 4) / 4
                    max_fix_lon = round(max_fix_lon * 4) / 4
                    if min_fix_lon <= 0.0:
                        min_fix_lon = 0.0
                    if max_fix_lon >= 360.0:
                        max_fix_lon = 359.75
                    print(sliceTime)
                    print(min_fix_lon)
                    print(max_fix_lon)
                    print(min_lat)
                    print(max_lat)
                    data = ds.sel(time=slice(sliceTime,sliceTime),longitude=slice(np.where(ds.lon == min_fix_lon)[0][0], np.where(ds.lon == max_fix_lon)[0][0]), latitude=slice(np.where(ds.lat == min_lat)[0][0], np.where(ds.lat == max_lat)[0][0]))
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
                                #OSCAR longitudes are 0-360; adjust back to +/- 180
                                fixLon = ((thisLon + 180) % 360) - 180
                                indices.append([thisLat, fixLon, data[variable].values[t][i][j]])
                        variableData.append(indices) 
            variableData = np.asarray(variableData)
            print(variableData)
            maskGroup.InputFiles = ','.join(oscarFileList)

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
                data.LongName = variableAttrs['standard_name']
                data.Units = variableAttrs['units']
                data.FillValue = -999.0

        lastMaskGroupName = maskGroupName

    ncFile.close()

    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = ncFilename
    uploadInfo['type'] = 'curated subset'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    jsonFilename = getCurationHierarchy(jobID, chunkID, curationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'curated hierarchy'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)
    closeDB(db)

    return
