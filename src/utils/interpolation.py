import scipy
import datetime
import numpy as np
import pandas as pd
import scipy.interpolate
import boto3
import xarray as xr
import s3fs
import json

from netCDF4 import Dataset
from database.connection import openDB, closeDB
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3Upload
from numpy import round
from utils.helpers import getInterpolationHierarchy
from utils import tos2ca_secrets


class MASK:

    def __init__(self, maskFile, timestamp_mask, anomaly_id):
        """
        Main function for the MASK class with gets the mask information
        for all the anomalies.
        :param maskFile: filename of the phdef mask file
        :type maksFile: str
        :param timestamp_mask: timestamp to read anomaly mask information from
        :type timestamp_mask: str
        :param anomaly_id: the specific anomaly to read mask infromation about
        :type anomaly_id: str
        """
        self.anomaly_num = anomaly_id

        fs = s3fs.S3FileSystem()
        rootgrp = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')

        mask_lat = rootgrp.lat.values[...]
        mask_lon = rootgrp.lon.values[...]

        rootgrp.close()
        rootgrp = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + timestamp_mask)

        mask_size = rootgrp.mask_indices.num_pixels.size
        mask_indices = rootgrp.mask_indices.values[...]

        rootgrp.close()

        lat_index = mask_indices[:, 0]
        lon_index = mask_indices[:, 1]
        self.id_event = mask_indices[:, 2]

        self.lon = np.ones((mask_size))*np.nan
        self.lat = np.ones((mask_size))*np.nan

        for ii in np.arange(0, mask_size):
            self.lon[ii] = mask_lon[lon_index[ii]]
            self.lat[ii] = mask_lat[lat_index[ii]]

    def find_anomaly(self, anomaly_id):
        """
        Function to get the mask information for a specific anomaly at a given timestamp
        :param anomaly_id: the anomaly of interest
        :type anomaly_id: str
        """
        iix = self.id_event == anomaly_id

        self.anomaly_lon = self.lon[iix]
        self.anomaly_lat = self.lat[iix]


class CURATED:

    def __init__(self, timestamp_curated, anomaly_id, grid_flag, curatedFile, variable):
        """
        This is the main function for the CURATED class.  This will prepare the curated data
        for interpolation onto the phdef temporal and spatial grid.
        :param timestamp_curated: the timestamp we're interpolating from the curated file
        :type timestamp_curated: str
        :param anomaly_id: the anomaly ID number we're interpolating
        :type anomaly_id: int
        :param grid_flag: a flag to let the class know how to handle the grid
        :type grid_flag: int
        :param curatedFile: filename of the curated file
        :type curatedFile: str
        :param variable: name of the variable the data is for
        :type variable: str
        """
        self.anomaly_id = anomaly_id
        fs = s3fs.S3FileSystem()
        rootgrp_curated = xr.open_dataset(fs.open(curatedFile, 'rb'), group=timestamp_curated + '/' + str(anomaly_id))
        self.longName = rootgrp_curated[variable].LongName
        self.units = rootgrp_curated[variable].Units
        self.variable = variable

        if (grid_flag == 1) | (grid_flag > 2) | (grid_flag == 0):
            tt1 = rootgrp_curated[variable].values[...]
            self.lat_array = tt1[1, :, 0]
            self.lon_array = tt1[1, :, 1]
            self.curated_var = tt1[1, :, 2]

        elif grid_flag == 2:

            index1 = 1
            index2 = 2

            tt1 = rootgrp_curated[variable].values[...]
            self.lat_array = tt1[index1, :, 0]
            self.lon_array = tt1[index1, :, 1]
            var_1 = tt1[index1, :, 2]
            var_2 = tt1[index2, :, 2]
            self.curated_var = (var_1 + var_2)/2

        self.delaunay1 = scipy.spatial.Delaunay(list(zip(self.lon_array, self.lat_array)))
        self.interpolator = scipy.interpolate.LinearNDInterpolator(self.delaunay1, self.curated_var)

    def setInterpolation(self, mask):
        """
        Function to interpoate latitude and longitude.
        :param mask: object from the MASK class
        :type mask: mask
        """

        lat = mask.anomaly_lat
        lon = mask.anomaly_lon

        self.var_interp = self.interpolator(lon, lat)

    def setCollocation(self, mask):
        """
        Function to collate data
        :param mask: object from the MASK class
        :type mask: mask
        """

        self.var_interp = -9999*np.ones(len(mask.anomaly_lat))

        for ii in np.arange(0, len(mask.anomaly_lat)):

            try:
                diff_lat = np.abs(self.lat_array - mask.anomaly_lat[ii])
                diff_lon = np.abs(self.lon_array - mask.anomaly_lon[ii])

                ix = np.where((diff_lat < 0.0001) & (diff_lon < 0.0001))
                self.var_interp[ii] = self.curated_var[ix]

            except:
                print("  Anomaly ID: ", mask.anomaly_num, " - no interpolation for point lat/lon: ", mask.anomaly_lat[ii], mask.anomaly_lon[ii])

    def setAveraging(self, mask, grid):
        """
        Function to average data.
        :param mask: object from the MASK class
        :type mask: mask
        :param grid: object from the GRID class
        :type grid: grid
        """

        self.var_interp = -9999*np.ones(len(mask.anomaly_lat))

        for ii in np.arange(0, len(mask.anomaly_lat)):

            try:
                ix = (self.lat_array >= mask.anomaly_lat[ii] - grid.curated_lat_grid) & (self.lat_array <= mask.anomaly_lat[ii] + grid.curated_lat_grid) & (self.lon_array >= mask.anomaly_lon[ii] - grid.curated_lon_grid) & (self.lon_array <= mask.anomaly_lon[ii] + grid.curated_lon_grid)  
                self.var_interp[ii] = np.nanmean(self.curated_var[ix])

            except:
                print("  Anomaly ID: ", mask.anomaly_num, " - no interpolation for point lat/lon: ", mask.anomaly_lat[ii], mask.anomaly_lon[ii])

    def write_interpolated_data(self, mask, grp1, time_str):
        """
        Function to write the interpolated data to a netCDF-4 file.
        :param mask: info from the MASK class
        :type mask: mask
        :param grp1: netCDF-4 group that is being written to
        :type grp1: netCDF4
        :param time_str: timestamp from the mask file
        :type time_str: str
        """

        tmp_data = -9999*np.ones([len(mask.anomaly_lat), 3])

        tmp_data[:, 0] = mask.anomaly_lat
        tmp_data[:, 1] = mask.anomaly_lon
        tmp_data[:, 2] = self.var_interp

        valid = (tmp_data[:, 2] >= -1000000000000000.0) & (tmp_data[:, 2] <= 1000000000000000.0)
        tmp_data[~valid, 2] = -9999

        grp2 = grp1.createGroup(str(self.anomaly_id))
        obs_dim = grp2.createDimension('observation', len(mask.anomaly_lat))   
        dpoints_dim = grp2.createDimension('data_point', 3)
        temp = grp2.createVariable(self.variable, np.float64, ('observation', 'data_point'), zlib=True)
        temp[...] = tmp_data
        temp.Description = "Each row is (lat,lon," + self.variable + "); the lat and lon represent the coordinates of the anomaly"
        temp.Time = time_str
        temp.LongName = self.longName
        temp.Units = self.units
        temp.FillValue = "-9999.0"

        data_column = tmp_data[:, 2]
        mask = data_column != -9999
        valid_data = data_column[mask]

        if valid_data.size > 0:

            percentiles = np.percentile(valid_data, [10, 25, 50, 75, 90])
            std_dev = np.std(valid_data)
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)

            temp.Min = '%.3f' % (min_val)
            temp.Max = '%.3f' % (max_val)
            temp.Mean = '%.3f' % (mean_val)
            temp.Std_dev = '%.3f' % (std_dev)
            temp.percentile_10 = '%.3f' % (percentiles[0])
            temp.percentile_25 = '%.3f' % (percentiles[1])
            temp.percentile_50 = '%.3f' % (percentiles[2])
            temp.percentile_75 = '%.3f' % (percentiles[3])
            temp.percentile_90 = '%.3f' % (percentiles[4])

        else:
            temp.Min = '-9999.0'
            temp.Max = '-9999.0'
            temp.Mean = '-9999.0'
            temp.Std_dev = '-9999.0'
            temp.percentile_10 = '-9999.0'
            temp.percentile_25 = '-9999.0'
            temp.percentile_50 = '-9999.0'
            temp.percentile_75 = '-9999.0'
            temp.percentile_90 = '-9999.0'


class GRID:

    def __init__(self, maskFile, t_mask_array, maskHierarchyInfo, curatedFile, t_curated_array, curatedHierarchyInfo):
        """
        Main function in GRID class to determine both the grid of the phdef and curation data sets.  
        Grid in this context is both temporal and spatial.  Everything will eventually be
        interpolated to the grid of the phdef (or mask) grid.
        :param maskFile: the phdef filename you want to open
        :type maskFile: str
        :param t_mask_array: the contents of the phdef hierarchy file
        :type t_mask_array: list
        :param maskHierarchyInfo: the mask file's hierarchy info
        :type maskHierarchyInfo: dict
        :param curatedFile: the curated filename you want to open
        :type curatedFile: str
        :param t_curated_array: the contents of the curation hierarchy file
        :type t_curated_array: list
        :param curatedHierarchyInfo: the curated hierarchy file's info
        :type curatedHierarchyInfo: dict
        """

        self.mask_deltat = np.timedelta64(pd.to_datetime(list(maskHierarchyInfo['masks'].keys())[1]) - pd.to_datetime(list(maskHierarchyInfo['masks'].keys())[0]), 'm')

        fs = s3fs.S3FileSystem()

        list_t = t_mask_array
        count = 0
        for iix in list_t:
            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + iix)
            dim = rootgrp_mask.mask_indices[...].data.shape[0]
            rootgrp_mask.close()
            if dim != 0:
                idx = count
                break
            count = count+1

        curated_group_1 = t_curated_array[idx]
        rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + iix)
        anomaly_1 = str(list(set(rootgrp_mask.mask_indices[...].data[:, 2]))[0])
        rootgrp_mask.close()
        variable = curatedHierarchyInfo[curated_group_1][anomaly_1][0]
        rootgrp_curated = xr.open_dataset(fs.open(curatedFile, 'rb'), group=curated_group_1 + '/' + anomaly_1)

        times_curated = (rootgrp_curated[variable].Times).split(",")
        curated_deltat = np.timedelta64((pd.to_datetime(times_curated[1]) - pd.to_datetime(times_curated[0])), 'm')

        if (self.mask_deltat >= curated_deltat):
            self.time_flag = 1
        elif (self.mask_deltat < curated_deltat):
            self.time_flag = 2

        rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')

        mask_lat_grid = abs(rootgrp_mask.lat[...][0] - rootgrp_mask.lat[...][1])
        mask_lon_grid = abs(rootgrp_mask.lon[...][0] - rootgrp_mask.lon[...][1])

        rootgrp_mask.close()

        curated_lat_array = np.sort(list(set(rootgrp_curated[variable][...][...].data[1, :, 0])))
        self.curated_lat_grid = abs(curated_lat_array[0] - curated_lat_array[1])

        curated_lon_array = np.sort(list(set(rootgrp_curated[variable][...][...].data[1, :, 1])))
        self.curated_lon_grid = abs(curated_lon_array[0] - curated_lon_array[1])

        if (round(mask_lat_grid, 3) == round(self.curated_lat_grid, 3)) & (round(mask_lon_grid, 3) == round(self.curated_lon_grid, 3)):

            self.spatial = 1

        elif (round(mask_lat_grid, 3) < round(self.curated_lat_grid, 3)) & (round(mask_lon_grid, 3) < round(self.curated_lon_grid, 3)):

            self.spatial = 2
        else:
            self.spatial = 3

        rootgrp_curated.close()


class SPATIAL:

    def __init__(self, maskFile, t_mask_array, curatedFile, t_curated_array, curatedHierarchyInfo):
        """
        Main function in SPATIAL class to determine both the grid of the phdef and curation data sets.  
        Grid in this context is only spatial.  Everything will eventually be
        interpolated to the grid of the phdef (or mask) grid.
        :param maskFile: the phdef filename you want to open
        :type maskFile: str
        :param t_mask_array: the contents of the phdef hierarchy file
        :type t_mask_array: list
        :param curatedFile: the curated filename you want to open
        :type curatedFile: str
        :param t_curated_array: the contents of the curation hierarchy file
        :type t_curated_array: list
        :param curatedHierarchyInfo: the curated hierarchy file's info
        :type curatedHierarchyInfo: dict
        """

        fs = s3fs.S3FileSystem()
        rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'))
        rootgrp_mask.close()

        rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + t_mask_array[0])
        
        curated_group_1 = str(list(curatedHierarchyInfo.keys())[0])
        anomaly_1 = str(list(set(rootgrp_mask.mask_indices[...].data[:, 2]))[0])
        print(curated_group_1)
        variable = curatedHierarchyInfo[curated_group_1][anomaly_1][0]
        
        rootgrp_curated = xr.open_dataset(fs.open(curatedFile, 'rb'), group=curated_group_1 + '/' + anomaly_1)

        rootgrp_mask.close()
        rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')

        mask_lat_grid = abs(rootgrp_mask.lat[...][0] - rootgrp_mask.lat[...][1])
        mask_lon_grid = abs(rootgrp_mask.lon[...][0] - rootgrp_mask.lon[...][1])

        rootgrp_mask.close()

        curated_lat_array = np.sort(list(set(rootgrp_curated[variable][...][...].data[1, :, 0])))
        self.curated_lat_grid = abs(curated_lat_array[0]-curated_lat_array[1])
        
        curated_lon_array = np.sort(list(set(rootgrp_curated[variable][...][...].data[1, :, 1])))
        self.curated_lon_grid = abs(curated_lon_array[0]-curated_lon_array[1])
    
        if (round(mask_lat_grid, 3) == round(self.curated_lat_grid, 3)) & (round(mask_lon_grid, 3) == round(self.curated_lon_grid, 3)):
            
            self.spatial = 1
            
        elif (round(mask_lat_grid, 3) < round(self.curated_lat_grid, 3)) & (round(mask_lon_grid, 3) < round(self.curated_lon_grid, 3)):
            
            self.spatial = 2
        else:
            self.spatial = 3

        rootgrp_curated.close()


def interpolator(jobID, chunkID):
    """
    This interpolator will take a curated data file and interpolate the data in space and time
    to the resolution of the data set that was used to generate the masks in the PhDef stage of TOS2CA
    :param jobID: the jobID of the data curation job you want to interpolate
    :type jobID: int
    :param chunkID: the chunk number for the job
    :type chunkID: int
    """
    db, cur = openDB()

    jobInfo = getJobInfo(cur, jobID, chunkID)[0]
    nChunks = jobInfo['nChunks']

    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    sql = 'SELECT location, type FROM output WHERE jobID=%s AND type IN ("masks", "hierarchy")'
    cur.execute(sql, (jobInfo['phdefJobID']))
    results = cur.fetchall()
    for result in results:
        if result['type'] == 'masks':
            maskFile = result['location']
        if result['type'] == 'hierarchy':
            maskHierarchyFile = result['location']

    cf = 's3://%s/%s/%s-%s-Curated-Data.nc4' % (bucketName, jobID, jobID, chunkID)
    ch = 's3://%s/%s/%s-%s-Curation-Hierarchy.json' % (bucketName, jobID, jobID, chunkID)
    sql = 'SELECT location, type, startDateTime FROM output WHERE jobID=%s AND location IN (%s, %s)'
    cur.execute(sql, (jobID, cf, ch))
    results = cur.fetchall()
    for result in results:
        if result['type'] == 'curated subset':
            curatedFile = result['location']
        if result['type'] == 'curated hierarchy':
            curatedHierarchyFile = result['location']
        startDateTime = result['startDateTime']

    s3 = boto3.resource('s3')
    maskHierarchyFileParts = maskHierarchyFile.split('/')
    content_object = s3.Object(maskHierarchyFileParts[2], '/'.join(maskHierarchyFileParts[3:]))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    maskHierarchyInfo = json.loads(file_content)
    curatedHierarchyFileParts = curatedHierarchyFile.split('/')
    content_object = s3.Object(curatedHierarchyFileParts[2], '/'.join(curatedHierarchyFileParts[3:]))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    curatedHierarchyInfo = json.loads(file_content)

    fs = s3fs.S3FileSystem()
    rootgrp_curated = xr.open_dataset(fs.open(curatedFile, 'rb'))

    # initialize output filename and global metadata
    output_file = '/data/tmp/%s-%s-Interpolated-Data.nc4' % (jobID, chunkID)
    interp_file = Dataset(output_file, 'w', format='NETCDF4')
    interp_file.Variable = rootgrp_curated.Variable
    interp_file.Dataset = rootgrp_curated.Dataset
    interp_file.Units = rootgrp_curated.Units
    interp_file.References = rootgrp_curated.References
    interp_file.Project = rootgrp_curated.Project
    interp_file.Institution = rootgrp_curated.Institution
    interp_file.ProductionTime = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    interp_file.PhDefJobID = rootgrp_curated.PhDefJobID
    interp_file.ProductInfo = rootgrp_curated.ProductInfo
    interp_file.FileFormat = rootgrp_curated.FileFormat

    variable = rootgrp_curated.Variable

    rootgrp_curated.close()
    rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')

    #create the navigation group within the interpolated file
    grp_navigation = interp_file.createGroup('navigation')
    grp_navigation.createDimension('lat', len(rootgrp_mask.lat))
    grp_navigation.createDimension('lon', len(rootgrp_mask.lon))

    temp_lat = grp_navigation.createVariable('lat', np.float32, ('lat'), zlib=True)
    temp_lon = grp_navigation.createVariable('lon', np.float32, ('lon'), zlib=True)
    temp_lat[...] = rootgrp_mask.lat.values
    temp_lon[...] = rootgrp_mask.lon.values

    rootgrp_mask.close()

    t_mask_array = list(maskHierarchyInfo['masks'].keys())[:]
    t_curated_array = list(curatedHierarchyInfo.keys())[:-1]
    t_lookup_array = []

    for thisTime in t_curated_array:
        ds = xr.open_dataset(fs.open(curatedFile, 'rb'), group=thisTime)
        t_lookup_array.append(ds.MaskTime)
    t_mask_array = t_lookup_array

    t_mask_array_name = t_mask_array[:]
    if chunkID == 1:
        t_mask_array[0] = t_mask_array[1]
    if chunkID == nChunks:
        t_mask_array[-1] = t_mask_array[-2]

    try:
        grid = GRID(maskFile, t_mask_array, maskHierarchyInfo, curatedFile, t_curated_array, curatedHierarchyInfo)
        scenario = grid.time_flag  # 1: mask time grid >= curated time grid / 2: mask time grid < curated time grid / 3: no anomalies
        spatial_grid_flag = grid.spatial  # 1: same grid / 2: mask grid < curated grid / 3: mask grid > curated grid / 4: no grid needed
    except UnboundLocalError:
        scenario = 3
        spatial_grid_flag = 4

    interpolationHierarchy = {}

    closeDB(db)

    if (scenario == 1) & (spatial_grid_flag == 1):  # set Collocation

        idx_curated = 0

        #loop over mask timestamp
        for timestamp_mask in t_mask_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + timestamp_mask)

            print("processing mask timestamp: ", t_mask_array_name[idx_curated])
            timestamp_curated = t_curated_array[idx_curated]
            grp1 = interp_file.createGroup(t_mask_array_name[idx_curated])
            interpolationHierarchy[t_mask_array_name[idx_curated]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            rootgrp_mask.close()

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, timestamp_mask, anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, scenario, curatedFile, variable)
                    curated.setCollocation(mask)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask)
                    interpolationHierarchy[t_mask_array_name[idx_curated]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

            idx_curated = idx_curated + 1

            rootgrp_mask.close()

    elif (scenario == 1) & (spatial_grid_flag == 2):  # set Interpolation

        idx_curated = 0

        #loop over mask timestamp
        for timestamp_mask in t_mask_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + timestamp_mask)

            print("processing timestamp: ", t_mask_array_name[idx_curated])
            timestamp_curated = t_curated_array[idx_curated]
            grp1 = interp_file.createGroup(t_mask_array_name[idx_curated])
            interpolationHierarchy[t_mask_array_name[idx_curated]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, timestamp_mask, anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, scenario, curatedFile, variable)
                    curated.setInterpolation(mask)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask)
                    interpolationHierarchy[t_mask_array_name[idx_curated]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

            idx_curated = idx_curated + 1

            rootgrp_mask.close()

    elif scenario == 2:

        idx_curated = 0

        for timestamp_mask in t_mask_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + timestamp_mask)

            print("processing timestamp: ", t_mask_array_name[idx_curated])
            t_flag = int(np.fromstring(t_curated_array[idx_curated], dtype=int, sep='-')[1])
            timestamp_curated = t_curated_array[idx_curated]
            grp1 = interp_file.createGroup(t_mask_array_name[idx_curated])
            interpolationHierarchy[t_mask_array_name[idx_curated]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, timestamp_mask, anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, t_flag, curatedFile, variable)
                    curated.setInterpolation(mask)
                    curated.write_interpolated_data(mask, grp1, t_mask_array_name[idx_curated])
                    interpolationHierarchy[t_mask_array_name[idx_curated]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

            idx_curated = idx_curated + 1

            rootgrp_mask.close()

    elif (scenario == 1) & (spatial_grid_flag == 3):  # set Averaging
 
        idx_curated = 0

        for timestamp_mask in t_mask_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + timestamp_mask)

            print("processing timestamp: ", t_mask_array_name[idx_curated])
            timestamp_curated = t_curated_array[idx_curated]
            grp1 = interp_file.createGroup(t_mask_array_name[idx_curated])
            interpolationHierarchy[t_mask_array_name[idx_curated]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, timestamp_mask, anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, scenario, curatedFile, variable)
                    curated.setAveraging(mask, grid)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask) 
                    interpolationHierarchy[t_mask_array_name[idx_curated]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

            idx_curated = idx_curated + 1

            rootgrp_mask.close()

    elif (scenario == 3) & (spatial_grid_flag == 4):

        for timestamp_mask in t_mask_array:
            print("processing timestamp: ", timestamp_mask)
            grp1 = interp_file.createGroup(timestamp_mask)
            interpolationHierarchy[timestamp_mask] = {}

    else:
        exit('Unable to determine grid.')

    print('Interpolation complete')
    interp_file.close()

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = output_file
    uploadInfo['type'] = 'interpolated subset'
    uploadInfo['startDateTime'] = startDateTime
    s3Upload(jobID, uploadInfo, bucketName, db, cur) 
    updateStatus(db, cur, jobID, "complete")

    jsonFilename = getInterpolationHierarchy(jobID, chunkID, interpolationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'interpolated hierarchy'
    uploadInfo['startDateTime'] = startDateTime
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    closeDB(db)

    return


def interpolator_stationary(jobID, chunkID):
    """
    This interpolator will take a curated data file and interpolate the data in space
    to the resolution of the data set that was used to generate the masks in the PhDef stage of TOS2CA
    :param jobID: the jobID of the data curation job you want to interpolate
    :type jobID: int
    :param chunkID: the chunk number of the job
    :type chunkID: int
    """
    db, cur = openDB()

    jobInfo = getJobInfo(cur, jobID)[0]

    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    sql = 'SELECT location, type FROM output WHERE jobID=%s AND type IN ("masks", "hierarchy")'
    cur.execute(sql, (jobInfo['phdefJobID']))
    results = cur.fetchall()
    for result in results:
        if result['type'] == 'masks':
            maskFile = result['location']
        if result['type'] == 'hierarchy':
            maskHierarchyFile = result['location']

    cf = 's3://%s/%s/%s-%s-Curated-Data.nc4' % (bucketName, jobID, jobID, chunkID)
    ch = 's3://%s/%s/%s-%s-Curation-Hierarchy.json' % (bucketName, jobID, jobID, chunkID)
    sql = 'SELECT location, type, startDateTime FROM output WHERE jobID=%s AND location IN (%s, %s)'
    cur.execute(sql, (jobID, cf, ch))
    results = cur.fetchall()
    for result in results:
        if result['type'] == 'curated subset':
            curatedFile = result['location']
        if result['type'] == 'curated hierarchy':
            curatedHierarchyFile = result['location']
        startDateTime = result['startDateTime']

    s3 = boto3.resource('s3')
    maskHierarchyFileParts = maskHierarchyFile.split('/')
    content_object = s3.Object(maskHierarchyFileParts[2], '/'.join(maskHierarchyFileParts[3:]))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    maskHierarchyInfo = json.loads(file_content)
    timestamp_mask = list(maskHierarchyInfo['masks'].keys())[0]
    curatedHierarchyFileParts = curatedHierarchyFile.split('/')
    content_object = s3.Object(curatedHierarchyFileParts[2], '/'.join(curatedHierarchyFileParts[3:]))
    file_content = content_object.get()['Body'].read().decode('utf-8')
    curatedHierarchyInfo = json.loads(file_content)
    
    fs = s3fs.S3FileSystem()
    rootgrp_curated = xr.open_dataset(fs.open(curatedFile, 'rb'))

    # initialize output filename and global metadata
    output_file = '/data/tmp/%s-%s-Interpolated-Data.nc4' % (jobID, chunkID)
    interp_file = Dataset(output_file, 'w', format='NETCDF4')
    interp_file.Variable = rootgrp_curated.Variable
    interp_file.Dataset = rootgrp_curated.Dataset
    interp_file.Units = rootgrp_curated.Units
    interp_file.References = rootgrp_curated.References
    interp_file.Project = rootgrp_curated.Project
    interp_file.Institution = rootgrp_curated.Institution
    interp_file.ProductionTime = str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    interp_file.PhDefJobID = rootgrp_curated.PhDefJobID
    interp_file.ProductInfo = rootgrp_curated.ProductInfo
    interp_file.FileFormat = rootgrp_curated.FileFormat

    variable = rootgrp_curated.Variable

    rootgrp_curated.close()
    rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='navigation')

    #create the navigation group within the interpolated file
    grp_navigation = interp_file.createGroup('navigation')
    grp_navigation.createDimension('lat', len(rootgrp_mask.lat))   
    grp_navigation.createDimension('lon', len(rootgrp_mask.lon))   

    temp_lat = grp_navigation.createVariable('lat',np.float32,('lat'),zlib=True)
    temp_lon = grp_navigation.createVariable('lon',np.float32,('lon'),zlib=True)
    temp_lat[...] = rootgrp_mask.lat.values
    temp_lon[...] = rootgrp_mask.lon.values

    rootgrp_mask.close()

    t_mask_array = list(maskHierarchyInfo['masks'].keys())[:]
    t_curated_array = list(curatedHierarchyInfo.keys())[:-1]

    try:
        grid = SPATIAL(maskFile, t_mask_array, curatedFile, t_curated_array, curatedHierarchyInfo)
        spatial_grid_flag = grid.spatial  # 1: same grid / 2: mask grid < curated grid / 3: mask grid > curated grid / 4: no grid needed
    except UnboundLocalError:
        spatial_grid_flag = 4

    interpolationHierarchy = {}

    closeDB(db)

    if spatial_grid_flag == 1:  # set Collocation

        #loop over mask timestamp
        for timestamp_mask in t_curated_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + t_mask_array[0])

            print("processing mask timestamp: ", timestamp_mask)
            timestamp_curated = timestamp_mask
            grp1 = interp_file.createGroup(timestamp_mask[:-4])
            interpolationHierarchy[timestamp_mask[:-4]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            rootgrp_mask.close()

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, t_mask_array[0], anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, 0, curatedFile, variable)
                    curated.setCollocation(mask)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask)
                    interpolationHierarchy[timestamp_mask[:-4]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)


            rootgrp_mask.close()

    elif spatial_grid_flag == 2:  # set Interpolation

        #loop over mask timestamp
        for timestamp_mask in t_curated_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + t_mask_array[0])

            print("processing timestamp: ", timestamp_mask)
            timestamp_curated = timestamp_mask
            grp1 = interp_file.createGroup(timestamp_mask[:-4])
            interpolationHierarchy[timestamp_mask[:-4]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            #loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, t_mask_array[0], anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, 0, curatedFile, variable)
                    curated.setInterpolation(mask)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask)
                    interpolationHierarchy[timestamp_mask[:-4]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

    elif spatial_grid_flag == 3:  # set Averaging

        for timestamp_mask in t_curated_array:

            rootgrp_mask = xr.open_dataset(fs.open(maskFile, 'rb'), group='masks/' + t_mask_array[0])

            print("processing timestamp: ", timestamp_mask)
            timestamp_curated = timestamp_mask
            grp1 = interp_file.createGroup(timestamp_mask[:-4])
            interpolationHierarchy[timestamp_mask[:-4]] = {}
            anomaly_num = np.sort(list(set(rootgrp_mask.mask_indices.values[...][:, 2])))

            # loop over anomaly id
            for anomaly_id in anomaly_num:

                try:
                    mask = MASK(maskFile, t_mask_array[0], anomaly_id)
                    mask.find_anomaly(anomaly_id)
                    curated = CURATED(timestamp_curated, anomaly_id, 0, curatedFile, variable)
                    curated.setAveraging(mask, grid)
                    curated.write_interpolated_data(mask, grp1, timestamp_mask) 
                    interpolationHierarchy[timestamp_mask[:-4]][str(anomaly_id)] = [variable]

                except:
                    print("Error in anomaly ID: ", anomaly_id)

            rootgrp_mask.close()

    elif spatial_grid_flag == 4:

        for timestamp_mask in t_curated_array:
            print("processing timestamp: ", timestamp_mask)
            grp1 = interp_file.createGroup(timestamp_mask[:-4])
            interpolationHierarchy[timestamp_mask[:-4]] = {}

    else:
        exit('Unable to determine grid.')

    print('Interpolation complete')
    interp_file.close()

    db, cur = openDB()

    uploadInfo = {}
    uploadInfo['filename'] = output_file
    uploadInfo['type'] = 'interpolated subset'
    uploadInfo['startDateTime'] = startDateTime
    s3Upload(jobID, uploadInfo, bucketName, db, cur) 
    updateStatus(db, cur, jobID, "complete")

    jsonFilename = getInterpolationHierarchy(jobID, chunkID, interpolationHierarchy)
    uploadInfo = {}
    uploadInfo['filename'] = jsonFilename
    uploadInfo['type'] = 'interpolated hierarchy'
    uploadInfo['startDateTime'] = startDateTime
    s3Upload(jobID, uploadInfo, bucketName, db, cur)

    closeDB(db)

    return
