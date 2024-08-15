import json
import numpy as np 
import s3fs
import shapely.wkt as wkt
import xarray as xr
import netCDF4 as nc

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import OrderedDict as ODict
from database.connection import openDB, closeDB, openCache
from database.elasticache import setData
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3GetTemporaryCredentials, s3Upload, checkReauth
from utils.helpers import get_json, pushBox, getCurationHierarchy
from shapely.geometry import MultiPoint

merra2 = {
    "M2I1NXINT_5.12.4" : "{year}/{month}/MERRA2_{stream}.inst1_2d_int_Nx.{year}{month}{day}.nc4",
    "M2IMNXINT_5.12.4" : "{year}/MERRA2_{stream}.instM_2d_int_Nx.{year}{month}.nc4",
    "M2TMNXLND_5.12.4" : "{year}/MERRA2_{stream}.tavgM_2d_lnd_Nx.{year}{month}.nc4",
    "M2I1NXLFO_5.12.4" : "{year}/{month}/MERRA2_{stream}.inst1_2d_lfo_Nx.{year}{month}{day}.nc4",
    "M2IMNXLFO_5.12.4" : "{year}/MERRA2_{stream}.instM_2d_lfo_Nx.{year}{month}.nc4",
    "M2I1NXASM_5.12.4" : "{year}/{month}/MERRA2_{stream}.inst1_2d_asm_Nx.{year}{month}{day}.nc4",
    "M2IMNXASM_5.12.4" : "{year}/MERRA2_{stream}.instM_2d_asm_Nx.{year}{month}.nc4",
    "M2I3NXGAS_5.12.4" : "{year}/{month}/MERRA2_{stream}.inst3_2d_gas_Nx.{year}{month}{day}.nc4",
    "M2SDNXSLV_5.12.4" : "{year}/{month}/MERRA2_{stream}.statD_2d_slv_Nx.{year}{month}{day}.nc4",
    "M2I3NVCHM_5.12.4" : "{year}/{month}/MERRA2_{stream}.inst3_3d_chm_Nv.{year}{month}{day}.nc4",
    "M2T1NXSLV_5.12.4" : "{year}/{month}/MERRA2_{stream}.tavg1_2d_slv_Nx.{year}{month}{day}.nc4",
    "M2SMNXEDI_2" : "{year}/MERRA2.statM_2d_edi_Nx.v2_1.{year}{month}.nc4"
}
streams = [100,200,300,400]

def getFileList(fs_s3, location, jobInfo, lastDayIsOverlap=False):
    """
    :param fs_s3: the fs_s3 filesystem object
    :type location: object
    :param location: the name of the DAAC
    :type location: str
    :param dataset: name of the dataset
    :type dataset: str
    :param startDate: start date time being searched
    :type startDate: datetime
    :param endDate: end date time being searched
    :type endDate: datetime
    :param creds: DAAC S3 credentials
    :type creds: dict
    :param jobType: phdef or curation
    :type jobType: str
    :param lastDayIsOverlap: a boolean to see if the last date should be treated as a chunk (first file only)
    :return files: list of filename to be processed
    :rtype files: list
    """
    dataset   = jobInfo['dataset']
    startDate = jobInfo['startDate']
    endDate   = jobInfo['endDate']
    jobType   = jobInfo['stage']

    pattern = merra2[dataset]
    files = []

    if jobType == 'phdef':
        currentDate = startDate
        pattern = merra2[dataset]
        files = []
        
        while currentDate <= endDate:
            year = currentDate.strftime("%Y")
            month = currentDate.strftime("%m")
            pattern = merra2[dataset]
            for stream in streams:
                if dataset == 'M2I1NXINT_5.12.4' or dataset == 'M2I1NXLFO_5.12.4' or dataset == 'M2SDNXSLV_5.12.4' or dataset == 'M2I1NXASM_5.12.4' or dataset == 'M2I3NXGAS_5.12.4' or dataset == 'M2I3NVCHM_5.12.4':
                    
                    day = currentDate.strftime("%d")
                    if "{stream}" in pattern:
                        name = pattern.format(year=year, month=month, day=day, stream=stream)
                    else:
                        name = pattern.format(year=year, month=month, day=day)

                else:
                    if "{stream}" in pattern:
                        name = pattern.format(year=year, month=month, stream=stream)
                    else:
                        name = pattern.format(year=year, month=month)

                fullPath = f"{location}{name}"
                g = fs_s3.glob(fullPath)
                for gf in g:
                    files.append(gf)
                    if(lastDayIsOverlap and currentDate == endDate): # Only want first file of last day in a chunking situation
                        break
            currentDate += timedelta(days=1)
    elif jobType == 'curation':
        if dataset == 'M2I1NXINT_5.12.4' or dataset == 'M2I1NXLFO_5.12.4' or dataset == 'M2T1NXSLV_5.12.4':
            startDate = startDate - timedelta(days=1)
            endDate = endDate + timedelta(days=1)
            timeDelta = endDate - startDate
            days = []
            for i in range(timeDelta.days + 1):
                day = startDate + timedelta(days=i)
                days.append(day)
            for currentDate in days:
                year = currentDate.strftime("%Y")
                month = currentDate.strftime("%m")
                day = currentDate.strftime("%d")
                if "{stream}" in pattern:
                    name = pattern.format(year=year, month=month, day=day, stream='*')
                else:
                    name = pattern.format(year=year, month=month, day=day)
                fullPath = f"{location}{name}"
                g = fs_s3.glob(fullPath)
                for gf in g:
                    files.append(gf)
                    if(lastDayIsOverlap and currentDate == days[-1]):
                        break
        else:
            startDate = startDate - relativedelta(months=1)
            endDate = endDate + relativedelta(months=1)
            timeDelta = endDate - startDate
            days = []
            for i in range(timeDelta.days + 1):
                day = startDate + timedelta(days=i)
                days.append(day.strftime('%Y-%m'))
            days = set(days)
            for currentDate in days:
                year = currentDate.split('-')[0]
                month = currentDate.split('-')[1]
                if "{stream}" in pattern:
                    name = pattern.format(year=year, month=month, stream='*')
                else:
                    name = pattern.format(year=year, month=month)
                fullPath = f"{location}{name}"
                g = fs_s3.glob(fullPath)
                for gf in g:
                    files.append(gf)

    return files

def rounder(t, interval):
    """
    Function to round to nearest timestemps for this dataset
    :param t: original date and time
    :type t: str
    :param interval: the internval of the data set (hourly, monthly, etc.), parsed from the filename
    :type interval: str
    :return timeIndex: three times (t-1, t, t+1)
    :rtype timeIndex: list
    """
    if 'inst' in interval:
        if '1' in interval:
            t = datetime.strptime(t, '%Y%m%d%H%M%S')
            minusT = (t - timedelta(hours=1)).replace(second=0, minute=0).strftime('%Y%m%d%H%M%S')
            originalT = t.replace(second=0, minute=0).strftime('%Y%m%d%H%M%S')
            plusT = (t + timedelta(hours=1)).replace(second=0, minute=0).strftime('%Y%m%d%H%M%S')
        if 'M' in interval:
            t = datetime.strptime(t, '%Y%m%d%H%M%S')
            minusT = (t - relativedelta(months=1)).replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
            originalT = t.replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
            plusT = (t + relativedelta(months=1)).replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
    elif 'tavg' in interval:
        if '1' in interval:
            t = datetime.strptime(t, '%Y%m%d%H%M%S')
            minusT = (t - timedelta(hours=1)).replace(second=0, minute=30).strftime('%Y%m%d%H%M%S')
            originalT = t.replace(second=0, minute=30).strftime('%Y%m%d%H%M%S')
            plusT = (t + timedelta(hours=1)).replace(second=0, minute=30).strftime('%Y%m%d%H%M%S')
        if 'M' in interval:
            t = datetime.strptime(t, '%Y%m%d%H%M%S')
            minusT = (t - relativedelta(months=1)).replace(second=0, minute=30, hour=0).strftime('%Y%m%d%H%M%S')
            originalT = t.replace(second=0, minute=30, hour=0).strftime('%Y%m%d%H%M%S')
            plusT = (t + relativedelta(months=1)).replace(second=0, minute=30, hour=0).strftime('%Y%m%d%H%M%S')
    elif 'stat' in interval:
        if 'M' in interval:
            t = datetime.strptime(t, '%Y%m%d%H%M%S')
            minusT = (t - relativedelta(months=1)).replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
            originalT = t.replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
            plusT = (t + relativedelta(months=1)).replace(second=0, minute=0, hour=0).strftime('%Y%m%d%H%M%S')
    else:
        exit()

    return [minusT, originalT, plusT]

def merra2_reader(jobID):
    """
    Function to submit MERRA-2 data from NASA Earthdata S3
    :param jobID: job ID to use to submit the request
    :type jobID: int
    """

    isChunk = '-' in str(jobID)

    db, cur = openDB()
    updateStatus(db, cur, jobID, "running")
    jobInfo = getJobInfo(cur, jobID)[0]
    r = openCache()
   
    # Retrieve credentials and location
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as phdef:
        info = json.load(phdef)
    daac = info[jobInfo['dataset']]['daac']
    creds = s3GetTemporaryCredentials(daac)
    location =info[jobInfo['dataset']]['location']
    
    lastDayIsOverlap = isChunk and (jobInfo['chunkID'] != jobInfo['nChunks']) 

    fs_s3 = s3fs.S3FileSystem(anon=False, 
                               key=creds['accessKeyId'], 
                               secret=creds['secretAccessKey'], 
                               token=creds['sessionToken'])

    files = getFileList(fs_s3, location, jobInfo, lastDayIsOverlap)
    nFiles = len(files)
    
    if len(files) == 0:
        print('No results found. Exiting...')
        updateStatus(db, cur, jobID, "error")
        exit(1)
    
    # Retrieve the job attributes 
    var = jobInfo['variable']
    coords = jobInfo['coords']
    polygon = wkt.loads(coords)
    min_lon, min_lat, max_lon, max_lat = polygon.bounds    

    data = {}
    read_count = 0
    # Package the results
    for i, filename in enumerate(files):
        newCredsNeeded = checkReauth(creds)
        if newCredsNeeded == 1:
            creds = s3GetTemporaryCredentials(daac)
        fs_s3 = s3fs.S3FileSystem(anon=False, 
                                key=creds['accessKeyId'], 
                               secret=creds['secretAccessKey'], 
                               token=creds['sessionToken'])
        # See if this is the last file in a chunking context
        readTime0Only = lastDayIsOverlap and i == (nFiles - 1)

        # Need additional guard to test for the correct stream <- Probably not needed in this case
        if fs_s3.exists(filename): # Probably can take this out if the filenames are already validated by s3
            with fs_s3.open(filename, mode='rb') as s3_file_obj:
                print('Reading %s' % filename)
                ds = xr.open_dataset(s3_file_obj)
                ds = ds.sel(lat=slice(min_lat,max_lat), lon=slice(min_lon,max_lon))
                if ds['lat'].values.size == 0 or ds['lon'].values.size == 0:
                    print('Warning: no valid lats or lons in file %s' % filename)
                    continue
                var_data = ds[var]
                if read_count == 0:
                    startTime = ds['time'].values[0].astype(str)[:-3]
                    start_time = datetime.strptime(startTime, '%Y-%m-%dT%H:%M:%S.%f')
                    data['name'] = jobInfo['dataset']
                    data['lat'] = np.asarray(ds.lat.values)
                    data['lon'] = np.asarray(ds.lon.values)
                    data['images'] = ODict()
                times = ds['time']
                for thisTime in times.values:
                    thisTimeString = str(thisTime)
                    timeString = datetime.strptime(thisTimeString[:-3], '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y%m%d%H%M')
                    data['images'][timeString] = np.asarray(var_data.sel(time=thisTime))
                    if readTime0Only:
                        break
                read_count += 1
        else: 
            continue
    
    if 'images' not in data.keys():
        exit('No data in bounds after file reads.')
    if len(data['images']) == 0:
        exit('No data in bounds after file reads.')

    setData(r, data, jobInfo, start_time, jobID)

    updateStatus(db, cur, jobID, "complete")

    closeDB(db)

    return

def merra2_curator(jobID):
    """
    Function to cruate data for MERRA-2, working on multiple datasets.
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
    quickInfo = {}
    quickInfo['startDate'] = startDate
    quickInfo['endDate']  = endDate
    quickInfo['dataset'] = dataset
    quickInfo['stage'] = 'curation'
    fs_s3 = s3fs.S3FileSystem(anon=False, 
                                    key=creds['accessKeyId'], 
                                    secret=creds['secretAccessKey'], 
                                    token=creds['sessionToken'])
    files = getFileList(fs_s3, location, quickInfo, False)
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
    #standard MERRA-2 data resolution?
    ncFile.DataResolution = '0.5 x 0.625'

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
        if dataset == 'M2SMNXEDI_2':
            fileTypeStep = 'statM'
        elif dataset == 'M2IMNXINT_5.12.4' or dataset == 'M2TMNXLND_5.12.4' or dataset == 'M2IMNXLFO_5.12.4' or dataset == 'M2IMNXASM_5.12.4':
            fileTypeStep = files[0].split('_')[2].split('.')[1]
        else:
            fileTypeStep = files[0].split('_')[1].split('.')[1]
        print(files[0])
        threeTimes = rounder(h, fileTypeStep)
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

            #Choose and read the right MERRA-2 file
            variableData = []
            merraFileList = []
            for thisTime in threeTimes:
                if timeStep == 'hourly':
                    mTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y%m%d')
                if timeStep == 'monthly':
                    mTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y%m')
                print(files)
                print(mTime)
                mFile = list(filter(lambda x: mTime in x, files))[0]
                print(mFile)
                merraFileList.append(mFile)
                fs_s3 = s3fs.S3FileSystem(anon=False, 
                                    key=creds['accessKeyId'], 
                                    secret=creds['secretAccessKey'], 
                                    token=creds['sessionToken'])
                
                with fs_s3.open(mFile, mode='rb') as s3_file_obj:
                    ds = xr.open_dataset(s3_file_obj)
                    #Expanding poloygon bounds to get grid cells that fall on the edges
                    min_lon, min_lat, max_lon, max_lat = pushBox(1.25, mp)
                    sliceTime = datetime.strptime(thisTime, '%Y%m%d%H%M%S').strftime('%Y-%m-%dT%H:%M:%S.%f000')
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
            variableData = np.asarray(variableData)
            print(variableData)
            maskGroup.InputFiles = ','.join(merraFileList)

            #write anomaly to the netCDF-4
            if len(variableData) != 0:
                observationDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('observation', None)
                dataPointDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('data_point', 3)
                timeStepDim = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createDimension('time_step', 3)
                data = ncFile[maskGroupName + '-' + inc][anomalyGroupName].createVariable(variable, 'f4', ('time_step', 'observation', 'data_point',), zlib=True, complevel=9)
                data[:] = np.asarray(variableData, dtype=np.float32)
                data.Description = 'Each row is (lat,lon,%s); the lat and lon represent the centroid point of the grid cell' % variable
                data.Times = ','.join(threeTimes)
                data.TimeIndexing = 'index 0 = t-1; index 1 = t; index 2 = t+1'
                data.Long_Name = variableAttrs['long_name']
                data.Units = variableAttrs['units']
                data.Fmissing_Value = variableAttrs['fmissing_value']
                data.Vmax = variableAttrs['vmax']
                data.Vmin = variableAttrs['vmin']
                data.Valid_Range = variableAttrs['valid_range']

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