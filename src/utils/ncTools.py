import s3fs
import xarray as xr
import netCDF4 as nc
import numpy as np

from database.connection import openDB, closeDB
from database.queries import deleteChunks
from utils.helpers import get_json, getInterpolationHierarchy, getCurationHierarchy
from utils.s3 import s3Upload, s3DeleteChunks
from collections import Counter
from utils import tos2ca_secrets


def combineCuratedFiles(jobID):
    """
    Function to take chunked curation files and stitch them into a single file
    :param jobID: jobID of the curation job
    :type jobID: int
    """
    db, cur = openDB()

    fileDict = {}
    sql = 'SELECT nChunks FROM jobs WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchone()
    nChunks = results['nChunks']
    chunkList = list(range(1, nChunks+1))
    for i in chunkList:
        fileDict[i] = {}

    sql = 'SELECT type, location FROM output WHERE jobID=%s AND type IN ("curated subset", "curated hierarchy")'
    cur.execute(sql, jobID)
    results = cur.fetchall()
    for result in results:
        try:
            chunkID = result['location'].split('/')[-1].split('-')[1]
            fileDict[int(chunkID)][result['type']] = result['location']
        except ValueError:
            exit('This is not a chunked file: %s' % result['location'])

    sql = 'SELECT MIN(startDate) AS startDate FROM chunks WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchall()
    startDate = results[0]['startDate']

    for i in fileDict:
        test = bool(fileDict[i])
        if test is False:
            exit('Missing file for chunk #%s' % i)

    closeDB(db)

    ncFilename = '/data/tmp/%s-Curated-Data.nc4' % jobID
    print(ncFilename)

    fs = s3fs.S3FileSystem()

    ds = xr.open_dataset(fs.open(fileDict[1]['curated subset'], 'rb'))
    ncFile = nc.Dataset(ncFilename, 'w', format='NETCDF4')
    metaKeys = list(ds.attrs.keys())
    for thisKey in metaKeys:
        ncFile.setncattr(thisKey, ds.attrs[thisKey])
    ds.close()

    ds = xr.open_dataset(fs.open(fileDict[1]['curated subset'], 'rb'), group='navigation')
    lons = np.asarray(ds['lon'][:])
    lats = np.asarray(ds['lat'][:])
    ds.close()

    #Write to netCDF-4 file
    navGroup = ncFile.createGroup('/navigation')
    latDim = ncFile['navigation'].createDimension('lat', len(lats))
    lonDim = ncFile['navigation'].createDimension('lon', len(lons))
    lat = ncFile['navigation'].createVariable('lat', 'f4', ('lat',), zlib=True, complevel=9)
    lon = ncFile['navigation'].createVariable('lon', 'f4', ('lon',), zlib=True, complevel=9)
    lat[:] = np.asarray(lats, dtype=np.float32)
    lon[:] = np.asarray(lons, dtype=np.float32)
    metaKeys = list(ds.attrs.keys())
    for thisKey in metaKeys:
        navGroup.setncattr(thisKey, ds.attrs[thisKey])
    curationHierarchy = {}

    timestampList = []
    for i in chunkList:
        j = get_json(fileDict[i]['curated hierarchy'])
        for t in j:
            if t != 'navigation':
                timestampList.append(t.split('-')[0])
    timestampCount = Counter(timestampList)

    incrementorDict = {}

    for i in timestampCount.keys():
        incrementorDict[i] = 1

    for i in chunkList:
        print('Processing chunk #%s' % i)
        fullHierarchy = get_json(fileDict[i]['curated hierarchy'])
        hInfo = list(fullHierarchy.keys())[:-1]
        for thisStamp in hInfo:
            timestamp = thisStamp.split('-')[0]
            if incrementorDict[timestamp] <= timestampCount[timestamp]:
                maskGroupName = '%s-%s' % (timestamp, incrementorDict[timestamp])
                incrementorDict[timestamp] += 1
            else:
                exit('Group timestamp incrementor is bad.')

            maskGroup = ncFile.createGroup(maskGroupName)
            ds = xr.open_dataset(fs.open(fileDict[i]['curated subset'], 'rb'), group=maskGroupName)
            metaKeys = list(ds.attrs.keys())
            for thisKey in metaKeys:
                maskGroup.setncattr(thisKey, ds.attrs[thisKey])
            curationHierarchy[maskGroupName] = {}

            for a in fullHierarchy[thisStamp]:
                ds = xr.open_dataset(fs.open(fileDict[i]['curated subset'], 'rb'), group=thisStamp + '/' + a )
                anomalyGroupName = a
                print('%s | Anomaly #%s' % (maskGroupName, a))
                anomalyGroup = ncFile[maskGroupName].createGroup(anomalyGroupName)
                variable = fullHierarchy[thisStamp][a][0]
                curationHierarchy[maskGroupName][anomalyGroupName] = [variable]

                observationDim = ncFile[maskGroupName][anomalyGroupName].createDimension('observation', None)
                dataPointDim = ncFile[maskGroupName][anomalyGroupName].createDimension('data_point', 3)
                timeStepDim = ncFile[maskGroupName][anomalyGroupName].createDimension('time_step', 3)
                data = ncFile[maskGroupName][anomalyGroupName].createVariable(variable, 'f4', ('time_step', 'observation', 'data_point',), zlib=True, complevel=9)
                data[:] = ds[variable].values
                metaKeys = list(ds[variable].attrs.keys())
                for thisKey in metaKeys:
                    data.setncattr(thisKey, ds[variable].attrs[thisKey])

                ds.close()

    ncFile.close()

    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = ncFilename
    uploadInfo['type'] = 'curated subset'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    jsonFilename = getCurationHierarchy(jobID, None, curationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'curated hierarchy'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    closeDB(db)

    return


def combineInterpolatedFiles(jobID):
    """
    Function to take chunked interpolated files and stitch them into a single file
    :param jobID: jobID of the curation job
    :type jobID: int
    """
    db, cur = openDB()

    fileDict = {}
    sql = 'SELECT nChunks FROM jobs WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchone()
    nChunks = results['nChunks']
    chunkList = list(range(1, nChunks+1))
    for i in chunkList:
        fileDict[i] = {}

    sql = 'SELECT type, location FROM output WHERE jobID=%s AND type IN ("interpolated subset", "interpolated hierarchy")'
    cur.execute(sql, jobID)
    results = cur.fetchall()
    for result in results:
        try:
            chunkID = result['location'].split('/')[-1].split('-')[1]
            fileDict[int(chunkID)][result['type']] = result['location']
        except ValueError:
            exit('This is not a chunked file: %s' % result['location'])

    sql = 'SELECT MIN(startDate) AS startDate FROM chunks WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchall()
    startDate = results[0]['startDate']

    for i in fileDict:
        test = bool(fileDict[i])
        if test is False:
            exit('Missing file for chunk #%s' % i)

    closeDB(db)

    ncFilename = '/data/tmp/%s-Interpolated-Data.nc4' % jobID
    print(ncFilename)

    fs = s3fs.S3FileSystem()

    ds = xr.open_dataset(fs.open(fileDict[1]['interpolated subset'], 'rb'))
    ncFile = nc.Dataset(ncFilename, 'w', format='NETCDF4')
    metaKeys = list(ds.attrs.keys())
    for thisKey in metaKeys:
        ncFile.setncattr(thisKey, ds.attrs[thisKey])
    ds.close()

    ds = xr.open_dataset(fs.open(fileDict[1]['interpolated subset'], 'rb'), group='navigation')
    lons = np.asarray(ds['lon'][:])
    lats = np.asarray(ds['lat'][:])
    ds.close()

    #Write to netCDF-4 file
    navGroup = ncFile.createGroup('/navigation')
    latDim = ncFile['navigation'].createDimension('lat', len(lats))
    lonDim = ncFile['navigation'].createDimension('lon', len(lons))
    lat = ncFile['navigation'].createVariable('lat', 'f4', ('lat',), zlib=True, complevel=9)
    lon = ncFile['navigation'].createVariable('lon', 'f4', ('lon',), zlib=True, complevel=9)
    lat[:] = np.asarray(lats, dtype=np.float32)
    lon[:] = np.asarray(lons, dtype=np.float32)
    metaKeys = list(ds.attrs.keys())
    for thisKey in metaKeys:
        navGroup.setncattr(thisKey, ds.attrs[thisKey])
    interpolationHierarchy = {}

    timestampList = []
    for i in chunkList:
        j = get_json(fileDict[i]['interpolated hierarchy'])
        for t in j:
            if t != 'navigation':
                timestampList.append(t.split('-')[0])
    timestampCount = Counter(timestampList)

    for i in chunkList:
        print('Processing chunk #%s' % i)
        fullHierarchy = get_json(fileDict[i]['interpolated hierarchy'])
        hInfo = list(fullHierarchy.keys())[:-1]
        for thisStamp in hInfo:
            timestamp = thisStamp.split('-')[0]
            maskGroupName = timestamp     
            if maskGroupName in ncFile.groups:
                print('Timestamp %s already added' % maskGroupName)
                continue
            else:
                maskGroup = ncFile.createGroup(maskGroupName)
                interpolationHierarchy[maskGroupName] = {}

                for a in fullHierarchy[thisStamp]:
                    ds = xr.open_dataset(fs.open(fileDict[i]['interpolated subset'], 'rb'), group=thisStamp + '/' + a )
                    anomalyGroupName = a
                    print('%s | Anomaly #%s' % (maskGroupName, a))
                    anomalyGroup = ncFile[maskGroupName].createGroup(anomalyGroupName)
                    variable = fullHierarchy[thisStamp][a][0]
                    interpolationHierarchy[maskGroupName][anomalyGroupName] = [variable]

                    observationDim = ncFile[maskGroupName][anomalyGroupName].createDimension('observation', None)
                    dataPointDim = ncFile[maskGroupName][anomalyGroupName].createDimension('data_point', 3)
                    data = ncFile[maskGroupName][anomalyGroupName].createVariable(variable, 'f4', ('observation', 'data_point',), zlib=True, complevel=9)
                    data[:] = ds[variable].values
                    metaKeys = list(ds[variable].attrs.keys())
                    for thisKey in metaKeys:
                        data.setncattr(thisKey, ds[variable].attrs[thisKey])

                    ds.close()

    ncFile.close()

    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = ncFilename
    uploadInfo['type'] = 'interpolated subset'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    jsonFilename = getInterpolationHierarchy(jobID, None, interpolationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'interpolated hierarchy'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    closeDB(db)

    return


def cleanUpChunks(jobID):
    """
    This script will run and cleanup all the 
    chunked files (curated and interpolated) for a
    given job.
    :param jobID: the jobID number
    :type jobID: int
    """
    db, cur = openDB()
    s3DeleteChunks(jobID, db, cur)
    deleteChunks(db, cur, jobID)
    closeDB(db)

    return
