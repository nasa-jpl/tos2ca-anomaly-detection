import json
import numpy as np 
import s3fs
import xarray as xr
import netCDF4 as nc

from datetime import datetime, timedelta
from database.connection import openDB, closeDB
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3Upload
from utils.helpers import get_json, pushBox, getCurationHierarchy, timerange, padTimestamps
from shapely.geometry import MultiPoint
from utils import tos2ca_secrets
from herbie import Herbie


def rounder(t):
    """
    Function to round to nearest timesteps for this dataset
    :param t: original date and time
    :type t: str
    :return timeIndex: three times (t-1, t, t+1)
    :rtype timeIndex: list
    """
    originalT = datetime.strptime(t, '%Y%m%d%H%M')          
    plusT = originalT + timedelta(hours=1)
    minusT = originalT - timedelta(hours=1)

    return [minusT.strftime('%Y%m%d%H%M%S'), originalT.strftime('%Y%m%d%H%M%S'), plusT.strftime('%Y%m%d%H%M%S')]


def hrrr_curator(jobID, chunkID):
    """
    Function to cruate data for HRRR using the Herbie library.
    This will download and read data through Herbie and subset it to the bounds of the anomaly.  It provides
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
    quickInfo = {}
    quickInfo['startDate'] = startDate
    quickInfo['endDate']  = endDate
    quickInfo['dataset'] = dataset
    quickInfo['stage'] = 'curation'
    closeDB(db)

    #Open the NetCDF-4 file; set global attributes
    ncFilename = '/data/tmp/%s-%s-Curated-Data.nc4' % (jobID, chunkID)
    ncFile = nc.Dataset(ncFilename, 'w', format='NETCDF4')
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
    ncFile.DataResolution = '3 km x 3 km'

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
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'hours','quantity':1}, first=True)
    elif chunkID == nChunks and nChunks > 1:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'hours','quantity':1}, last=True)
    elif nChunks == 1:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'hours','quantity':1}, first=True, last=True)
    else:
        newTimestamps = padTimestamps(hierarchyTimes, {'units':'hours','quantity':1})
    print(newTimestamps)
    for h in newTimestamps:
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
        maskData = np.zeros((latShape, lonShape), dtype=int)
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

            variableData = []
            hrrrFileList = []
            for thisTime in threeTimes:
                H = Herbie(datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S'), model='hrrr', fxx=1, product='sfc')
                H.download()
                gribFile = str(H.get_localFilePath())
                print(gribFile)
                hrrrFileList.append(gribFile)
                ds = xr.load_dataset(gribFile, engine="cfgrib", filter_by_keys={'typeOfLevel': 'surface', 'stepType':'instant'})
                min_lon, min_lat, max_lon, max_lat = pushBox(0.2, mp)

                lat_new = np.linspace(min_lat, max_lat, 100)
                lon_new = np.linspace(min_lon, max_lon, 100)
                lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
                fixLats = ds['latitude'].values
                fixLons = ((ds['longitude'].values + 180) % 360) - 180
                points = np.column_stack((fixLons.ravel(), fixLats.ravel()))
                values = ds[variable.lower()].values.ravel()            

                variableName = ds[variable.lower()].name
                variableAttrs = ds[variable.lower()].attrs
                globalAttrs = ds.attrs
                indices = []

                for i, thisPoint in enumerate(points):
                    indices.append([thisPoint[1], thisPoint[0], values[i]])
                variableData.append(indices) 
            variableData = np.asarray(variableData)
            print(variableData)
            maskGroup.InputFiles = ','.join(hrrrFileList)

            # write anomaly to the netCDF-4
            # not filtering for quality, seems to only be experimental quality filter recommendations
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
                data.FillValue = -9999.0

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


def hrrr_curator_stationary(jobID, chunkID):
    """
    Function to cruate data for HRRR using the Herbie library.
    This will download and read data through Herbie and subset it to the bounds of the anomaly.  It provides
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
    closeDB(db)

    #Open the NetCDF-4 file; set global attributes
    ncFilename = '/data/tmp/%s-%s-Curated-Data.nc4' % (jobID, chunkID)
    ncFile = nc.Dataset(ncFilename, 'w', format='NETCDF4')
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
    ncFile.DataResolution = '3 km x 3 km'

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
    h = list(hierarchyInfo['masks'])[0]

    ds = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + h)
    mask_indices = np.asarray(ds['mask_indices'][:])
    anomalies = []
    for a in ds.mask_indices:
        anomalies.append(a.values[2])
    anomalies = list(set(anomalies))
    anomalies.sort()
    ds.close()

    maskData=np.zeros((latShape, lonShape), dtype=int)
    thisAnomaly = anomalies[0]
    coords = []
    print('create the mask')
    for line in mask_indices:
        i,j,storm_id=line
        lon, lat = lons[j], lats[i]
        lon_idx = int((np.round(float(lon)/lon_res) * lon_res-x[0][0])/lon_res)
        lat_idx = int((np.round(float(lat)/lat_res)* lat_res-y[0][0])/lat_res)
        maskData[lat_idx][lon_idx] = int(storm_id)
        if storm_id == thisAnomaly:
            coords.append([lon, lat])
    mp = MultiPoint(coords)

    dateRange = timerange(startDate, endDate, 'H')

    indices = [[],[],[]]
    for thisTime in dateRange:
        thisTime = thisTime.strftime('%Y%m%d%H%M')
        print(thisTime)
        #Read the mask file
        threeTimes = rounder(thisTime)
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
        maskGroup.MaskTime = threeTimes[1]
        maskGroup.MaskFileName = 'Anomaly masks from %s' % maskFile
        #create anomaly group for netCDF-4 file using the anomalyID 
        anomalyGroupName = str(thisAnomaly)
        print('Creating NetCDF-4 group ' + anomalyGroupName)
        anomalyGroup = ncFile[maskGroupName + '-' + inc].createGroup(anomalyGroupName)
        curationHierarchy[maskGroupName + '-' + inc][anomalyGroupName] = [variable]

        #Choose and read the HRRR file
        variableData = []
        hrrrFileList = []
        for thisTime in threeTimes:
            H = Herbie(datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S'), model='hrrr', fxx=1, product='sfc')
            H.download()
            gribFile = str(H.get_localFilePath())
            print(gribFile)
            hrrrFileList.append(gribFile)
            ds = xr.load_dataset(gribFile, engine="cfgrib", filter_by_keys={'typeOfLevel': 'surface', 'stepType':'instant'})
            min_lon, min_lat, max_lon, max_lat = pushBox(0.2, mp)

            lat_new = np.linspace(min_lat, max_lat, 100)
            lon_new = np.linspace(min_lon, max_lon, 100)
            lon_new_grid, lat_new_grid = np.meshgrid(lon_new, lat_new)
            lats = ds['latitude'].values
            lons = ((ds['longitude'].values + 180) % 360) - 180
            points = np.column_stack((lons.ravel(), lats.ravel()))
            values = ds[variable.lower()].values.ravel()            

            variableName = ds[variable.lower()].name
            variableAttrs = ds[variable.lower()].attrs
            globalAttrs = ds.attrs
            indices = []

            for i, thisPoint in enumerate(points):
                indices.append([thisPoint[1], thisPoint[0], values[i]])
            variableData.append(indices) 
        variableData = np.asarray(variableData)
        print(variableData)
        maskGroup.InputFiles = ','.join(hrrrFileList)

        # write anomaly to the netCDF-4
        # not filtering for quality, seems to only be experimental quality filter recommendations
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
            data.FillValue = -9999.0

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