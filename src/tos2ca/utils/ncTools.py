import s3fs
import xarray as xr
import netCDF4 as nc
import numpy as np
import h5netcdf
import json

from database.connection import openDB, closeDB
from database.queries import deleteChunks
from utils.helpers import get_json, getInterpolationHierarchy, getCurationHierarchy
from utils.s3 import s3Upload, s3DeleteChunks
from collections import Counter
from utils import tos2ca_secrets


def copy_variable_streaming(var, outvar, chunk_size=100):
    """
    Copy variable data from xarray.DataArray to netCDF4.Variable in chunks.
    """
    shape = var.shape
    if len(shape) == 0:
        outvar[...] = var.values
        return

    n = shape[0]  # first dimension
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        slicer = (slice(start, stop),) + (slice(None),) * (len(shape) - 1)

        data_chunk = var[slicer].values  # lazy load only this slice
        outvar[slicer] = data_chunk


def copy_group_recursive(src, dst, chunk_size=100):
    """
    Recursively copy group structure, variables, and attributes
    from a netCDF4 group (src) to another (dst).
    """
    # 1. Copy dimensions
    for dname, dim in src.dimensions.items():
        if dname not in dst.dimensions:
            dst.createDimension(dname, (len(dim) if not dim.isunlimited() else None))

    # 2. Copy variables
    for vname, var in src.variables.items():
        if vname in dst.variables:
            continue
        outvar = dst.createVariable(vname, var.datatype, var.dimensions)
        outvar.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
        copy_variable_streaming(xr.DataArray(var[:]), outvar, chunk_size=chunk_size)

    # 3. Copy attributes
    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})

    # 4. Copy subgroups
    for subname, subgrp in src.groups.items():
        new_subgrp = dst.createGroup(subname)
        copy_group_recursive(subgrp, new_subgrp, chunk_size)


def group_to_dict_simple(filename):
    """
    Recursively convert a netCDF4.Group into dict of subgroups and variable names.
    """
    info = {}
    data = nc.Dataset(filename, 'r')
    for groupName in data.groups.keys():
        if groupName == 'navigation':
            info[groupName] = list(data[groupName].variables.keys())
        else:
            info[groupName] = {}
            if not data[groupName].groups.keys():
                continue
            for subgroupName in data[groupName].groups.keys():
                if not data[groupName][subgroupName].variables.keys():
                    continue
                for variable in data[groupName][subgroupName].variables.keys():
                    info[groupName][subgroupName] = [variable]

    return info


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
        print('Processing %s' % fileDict[i]['interpolated hierarchy'])
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


def mergeInterpolatedFiles(jobID):
    """
    Function to take chunked interpolated files and stitch them into a single file
    This is an alternative to combineInterpolatedFiles above.
    This should be used for larger jobs, probably shouldn't try to process more than
    one year's worth of data at a time.
    :param jobID: jobID of the curation job
    :type jobID: int
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")
    
    db, cur = openDB()

    sql = 'SELECT startDate FROM jobs WHERE jobID=(SELECT phdefJobID FROM jobs WHERE jobID=%s)'
    cur.execute(sql, jobID)
    results = cur.fetchone()
    startDate = results['startDate']

    sql = 'SELECT chunkID FROM chunks WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchall()
    print(results)
    chunkList = []
    for chunkID in results:
        chunkList.append(chunkID['chunkID'])

    fileList = []
    for chunkID in chunkList:
        sql = 'SELECT location FROM output WHERE jobID=%s AND location="s3://%s/%s/%s-%s-Interpolated-Data.nc4"' % (jobID, bucketName, jobID, jobID, chunkID)
        cur.execute(sql)
        results = cur.fetchone()
        if len(results) < 1:
            exit('Could not find a file for: %s' % chunkID)
        else:
            fileList.append(results['location'])

    closeDB(db)

    ncFilename = '/data/tmp/%s-Interpolated-Data.nc4' % (jobID)
    print(ncFilename)

    fs = s3fs.S3FileSystem()

    merged_groups = {}

    for path in fileList:
        with fs.open(path, "rb") as fobj:
            with h5netcdf.File(fobj, "r") as h5file:
                for gname in h5file.groups.keys():
                    if gname in merged_groups:
                        print(f"Skipping group '{gname}' in {path} (already loaded)")
                        continue
                    merged_groups[gname] = path

    print(f"Found {len(merged_groups)} unique top-level groups across {len(fileList)} files")

    with nc.Dataset(ncFilename, "w") as dst:
        for gname, src_path in merged_groups.items():
            print(f"Writing group: {gname} from {src_path}")
            with fs.open(src_path, "rb") as fobj:
                with nc.Dataset("inmemory", memory=fobj.read()) as src:
                    src_grp = src.groups[gname]
                    new_grp = dst.createGroup(gname)
                    copy_group_recursive(src_grp, new_grp, chunk_size=100)

    json_file = "/data/tmp/%s-Interpolation-Hierarchy.json" % (jobID)
    
    structure = group_to_dict_simple(ncFilename)

    with open(json_file, "w") as f:
        json.dump(structure, f)

    print(f"JSON structure written to {json_file}")

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = ncFilename
    uploadInfo['type'] = 'interpolated subset'
    uploadInfo['startDateTime'] = startDate
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    uploadInfo = {}
    uploadInfo['filename'] = json_file
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
