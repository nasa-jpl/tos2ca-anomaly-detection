"""
1. Read table of content e.g. 47-ForTraCC-TOC.json (Nanomalies)
2. Read output hierarchy, e.g. 47-ForTraCC-Mask-Output-Hierarchy.json
   This gives the netcdf output file group hierarchy. This is needed
   because we need to read the file remotely, on S3. For this purpose,
   we use `xarray` which does not provide means to get this hierarchy
   (unlike`netCDF4 which does, but is local). (Nmasks)
   Note: Nmasks can be smaller than Nanomalies, e.g. 96 < 585. In other
   words, there can be several anomalies per mask.
3. Get mask(s) indices from the netcdf output file e.g.,
   47-ForTraCC-Mask-Output.nc4: 0 <= mask_ix < Nmasks.
4. Read data curation dictionary e.g., tos2ca-data-collection-dictionary.json
   IMPORTANT: the location of this file is not fixed yet (07/19/2023).
5. Choose anomaly from from the table of content TOC file. Those anomalies
    must be choosen among the anomalies existing for this mask. 
6. Choose variable that must exist in the data curation dictionnary.
7. Subset the variable using the different masks from step 3.
"""
import os, json, sys, pprint
import s3fs
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from shapely import Point, MultiPoint, wkt, Polygon
from shapely.geometry import CAP_STYLE, JOIN_STYLE
import shapely as sp
import numpy as np
import time, re
import netCDF4 as nc
from collections import OrderedDict
import warnings
from utils.helpers import getCurationHierarchy

warnings.filterwarnings("ignore", category=xr.SerializationWarning)

utils_dir = '/data/code/anomaly-detection/src'
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
from database.connection import openDB, closeDB
from database.queries import getJobInfo, updateStatus
from utils.s3 import s3GetTemporaryCredentials, s3Upload
    
class Curation:
    def __init__(self):
        print("__init__()")
        # PO.DAAC server:
        self.expiration = None
        self.expiration_ft = "%Y-%m-%d %H:%M:%S%z"
        self.dt = timedelta(minutes=30) # 5)
        #
        self.data_col_dict = ('/data/code/data-dictionaries/'+
                              'tos2ca-data-collection-dictionary.json')

    @staticmethod    
    def print_line(c,n):
        """Print `n` times the character `c`."""
        print( ''.join( [c for i in range(n)] ) )


    @staticmethod
    def print_line2(c,n,pre='',post=''):
        """Print `n` times the character `c`."""
        print( pre + ''.join( [c for i in range(n)] ) + post )


    @staticmethod
    def get_json(filename):
        """
        """
        fs = s3fs.S3FileSystem()
        with fs.open(filename, 'r') as f:
            j = json.load(f)
        return j


    @staticmethod
    def load_json(filename):
        """load_json(filename)"""
        if os.path.isfile(filename):
            print(f"Load {filename}.")
            f = open(filename,'r')
            j = json.load(f)
            f.close()
            return j
        else:
            print('Error in load_json, file',filename,' does not exist.')
            return {}


    @staticmethod
    def save_json(json_struc,filename):
        if os.path.isfile(filename):
            print(f"{filename} already exists. Do not save.")
        else:
            print(f"Save {filename}.")
            f = open(filename,'w')
            json.dump(json_struc,f)
            f.close()


    @staticmethod
    def get_mask(ncfile,groupname,printQ=False):
        fs = s3fs.S3FileSystem()
        try:
            group = xr.open_dataset(fs.open(ncfile, 'rb'),group='masks/'+groupname)
            if printQ:
                print(f"successful x.open_dataset(fs.open({ncfile}, 'rb'),"
                    f"group={'masks/'+groupname})")
        except OSError as error:
            print(f"Error xr.open_dataset(fs.open({ncfile}, 'rb'),group={'masks/'+groupname})")
            print(error)
            group = None
        return group


    @staticmethod
    def get_curation_dictionary(curdicname):
        """
        Read the curation dictionary locally on tos2ca-dev1.
        Example: /data/code/data-dictionaries/tos2ca-data-collection-dictionary.json"
        """
        with open(curdicname,'r') as f:
            j = json.load(f)
        return j


    @staticmethod
    def anomaly_datetime(anomaly_date_time_string):
        """
        Transform anomaly time strings (from TOC file) in datetime.
        Example: {'name': 'Anomaly 1',
                'start_date': '202303010000',
                'end_date': '202303010000'}
        """
        return datetime.strptime(anomaly_date_time_string,"%Y%m%d%H%M%S")


    def get_PODAAC_credentials(self,printQ=False):
        if printQ:
            print(f"Get S3 credentials at {datetime.now()}")
        self.s3_creds = s3GetTemporaryCredentials('PO.DAAC')
        if printQ: 
            print(f"credentials:")
            pprint.pprint(self.s3_creds)

        if not self.s3_creds.keys() & {'accessKeyId','secretAccessKey','sessionToken','expiration'}:
            print('Error in get_PODAAC_credentials()')
        else:
            expiration = datetime.strptime(self.s3_creds['expiration'],self.expiration_ft)
            # expiration = datetime.fromisoformat(s3_creds['expiration'])
            self.expiration = expiration.replace(tzinfo=None)
            self.cred_time = datetime.now()


    def update_PODAAC_credentials(self,printQ=False):
        if self.expiration is None:
            self.get_PODAAC_credentials(printQ)
        elif self.cred_time + self.dt < datetime.now():
            self.get_PODAAC_credentials(printQ)
        # else: # do nothing


    def get_variable_files(self,var,printQ=False):
        """
        var['daac'] == 'PO.DAAC'
        """
        self.update_PODAAC_credentials(printQ)

        # /data/code/anomaly-detection/src/iolib/gpm.py:
        fs_s3 = s3fs.S3FileSystem(anon=False,
                                key    = self.s3_creds['accessKeyId'], 
                                secret = self.s3_creds['secretAccessKey'], 
                                token  = self.s3_creds['sessionToken'])
        
        files = fs_s3.glob(var['location']+"*.nc")
        return files

    
    @staticmethod
    def get_ASCAT_datetimes(ascatb_file):
        """
        * YYYYMMDD denotes the date of the first data in the file
        * HHMMSS denotes the time (UTC) of the first data in the file
        """
        fields = ascatb_file.split('_')
        return datetime.strptime(fields[1]+fields[2],"%Y%m%d%H%M%S")


    @staticmethod
    def search_files(start_date,end_date,var_dates,printQ=False):
        """
        Assume the `var_dates` list is sorted in ascending order of datetimes.
        """
        assert isinstance(start_date,datetime), "Error: start_date should be a datetime"
        assert isinstance(end_date,datetime), "Error: end_date should be a datetime"
        assert start_date <= end_date, "Error: start_date should be <= end_date"
        assert var_dates[0] <= start_date, "Error: var_dates[0] should be <= start_date"
        td = timedelta(seconds=6180) # ~ 1h43min
        assert end_date <= var_dates[-1]+td, "Error: end_date should be <= var_dates[-1]+tde"

        ps = pd.Series(var_dates)

        if start_date == end_date:
            if printQ: print(f"{start_date} == {end_date}")
            ixs = ixe = ps[ ps <= start_date ].index[-1].tolist()
            return ixs,ixe
        
        indices = ps[ ps.between( start_date, end_date ) ].index

        if indices.empty:
            ixs = ixe = ps[ ps <= start_date ].index[-1].tolist()
            return ixs,ixe

        indices = indices.to_list()
        if 0 < indices[0]:
            if printQ: print(f"0 < indices[0]: {indices[0]}")
            ixs = indices[0] - 1
        else:
            print(f"Error: 0 >= indices[0]: 0 >= {indices[0]}")

        ixe = indices[-1]
            
        return ixs,ixe


    @staticmethod
    def get_navigation(ncfile,printQ=False):
        fs = s3fs.S3FileSystem()
        try:
            group = xr.open_dataset(fs.open(ncfile, 'rb'),group='navigation')
            if printQ:
                print(f"successful x.open_dataset(fs.open({ncfile}, 'rb'),"
                    f"group={'navigation'})")
            lat = np.asarray( group['lat'][:] )
            lon = np.asarray( group['lon'][:] )
        except OSError as error:
            print(f"Error xr.open_dataset(fs.open({ncfile}, 'rb'),group={'navigation'})")
            print(error)
            (lat,lon) = (None,None)
        return lat,lon


    def get_group_names(self):
        """
        Mask-Output-Hierarchy file:
        mohfile = "s3://tos2ca-dev1/47/47-ForTraCC-Mask-Output-Hierarchy.json"
        moh = self.get_json(mohfile)
        """
        moh = self.get_json(self.hierarchyFile) # set line 1471
        return list( moh['masks'].keys() )


    @staticmethod
    def get_group_names_datetimes(groupnames):
        return [ datetime.strptime(gn,"%Y%m%d%H%M") for gn in groupnames ]


    def ascat_files_info(self,data_name,data_collection_dictionary):
        """
        Input
          data_name: 'ASCATB-L2-25km', 'ASCATB-L2-25km', ...
        """
        if data_name in data_collection_dictionary.keys():
            ascat = data_collection_dictionary.get(data_name)
            fs  = self.get_variable_files(ascat) # ~ 22s
            dts = [ self.get_ASCAT_datetimes(f) for f in fs  ]
            return fs,dts
        else:
            print(f"Error in ascat_files_info(): {data_name} is not ",end='')
            print("a key of the data collection dictionary.")
            return [],[]
        

    def ascatb_files_info(self,data_collection_dictionary):
        if 'ASCATB-L2-25km' in data_collection_dictionary.keys():
            ascatb = data_collection_dictionary.get('ASCATB-L2-25km')
            fs  = self.get_variable_files(ascatb) # ~ 22s
            dts = [ self.get_ASCAT_datetimes(f) for f in fs  ]
            return fs,dts
        else:
            print("Error in ascatb_files_info(): 'ASCATB-L2-25km' is not ",end='')
            print("a key of the data collection dictionary.")
            return [],[]


    def ascatc_files_info(self,data_collection_dictionary):
        if 'ASCATC-L2-25km' in data_collection_dictionary.keys():
            ascatc = data_collection_dictionary.get('ASCATC-L2-25km')
            fs  = self.get_variable_files(ascatc) # ~ 22s
            dts = [ self.get_ASCAT_datetimes(f) for f in fs  ]
            return fs,dts
        else:
            print("Error in ascatc_files_info(): 'ASCATC-L2-25km' is not ",end='')
            print("a key of the data collection dictionary.")
            return [],[]
        

    @staticmethod
    def data_in_mask(mlat,mlon,lat,lon,printQ=False,**kwargs):
        """
        Returns True if the geometries defined by the data coordinates
        (lat,lon) and by the mask points coordinates (mla,mlon) are within
        a given distance.
        Inputs
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput
            data_ixs: indices of data points that are within a distance
                the mask data. The distance `distance` is either given as
                an optional parameter with the keyword `distance` or is
                calculated as the minimum distance (in degrees) between all
                mask data points.
        """
        mp_mask = MultiPoint( [ Point(la,lo) for la,lo in zip(mlat,mlon) ] )
        mp_data = MultiPoint( [ Point(la,lo) for la,lo in zip(lat,lon) ] )
        dmin = sp.distance(mp_mask,mp_data)
        if printQ: print(f"mask-data distance: {dmin:.3f} [degree]")

        # minimum points spacing :
        dlat = mlat[1:] - mlat[:-1]
        dlat = dlat[dlat!=0]
        dlon = mlon[1:] - mlon[:-1]
        dlon = dlon[dlon!=0]
        data_ixs = []

        # (minimum) distance between the mask and data points:
        if 'distance' in kwargs.keys():
            d = kwargs['distance']
        else:
            # minimum distance between mask points:
            d = min( np.abs(dlat).min(), np.abs(dlon).min() )
            if printQ:
                print(f"minimum distance between mask points: {d:.3f} [degrees].")

        # Iterate through all data points and test if they are in the mask:
        # print(f"debug len(mp_data.geoms) = {len(mp_data.geoms)}")
        for i in range( len(mp_data.geoms) ):
            data_geom = sp.get_geometry(mp_data,i)
            if mp_mask.dwithin(data_geom,d):
                data_ixs.append(i)

        return data_ixs

    @staticmethod
    def data_in_mask_box(mlat,mlon,lat,lon,printQ=False,**kwargs):
        """
        Returns True if the geometries defined by the data coordinates
        (lat,lon) and by the mask points coordinates (mla,mlon) are within
        a given distance.
        Inputs
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
        Ouput
            data_ixs: indices of data points that are within 
                the mask data. 
        """
        mp_mask = MultiPoint( [ Point(la,lo) for la,lo in zip(mlat,mlon) ] )
        mp_data = MultiPoint( [ Point(la,lo) for la,lo in zip(lat,lon) ] )
        wktObj = mp_mask.wkt
        multi_point = wkt.loads(wktObj)
        interval = 5.0    
        kwargs = {"cap_style": CAP_STYLE.square, "join_style": JOIN_STYLE.mitre}
        poly = multi_point.buffer(interval/2, **kwargs).buffer(-interval/2, **kwargs)
        min_lon, min_lat, max_lon, max_lat = poly.bounds
        polygon = Polygon().from_bounds(min_lon - 0.23, min_lat - 0.23, max_lon + 0.23, max_lat + 0.23)
        data_ixs = []

        # Iterate through all data points and test if they are in the mask:
        # print(f"debug len(mp_data.geoms) = {len(mp_data.geoms)}")
        for i in range( len(mp_data.geoms) ):
            data_geom = sp.get_geometry(mp_data,i)
            if polygon.contains(data_geom):
                data_ixs.append(i)

        return data_ixs


    def get_ASCAT_data(self,filename):
        self.update_PODAAC_credentials(False)
        fs_s3 = s3fs.S3FileSystem(anon=False,
                                    key    = self.s3_creds['accessKeyId'], 
                                    secret = self.s3_creds['secretAccessKey'], 
                                    token  = self.s3_creds['sessionToken'])
        with fs_s3.open(filename,'rb') as f:
            print(f"f = fs_s3.open({filename},'rb')")
            data       = xr.open_dataset(f)
            time_ascat = np.asarray( data['time'][:] )
            wind_speed = np.asarray( data.get('wind_speed')[:] )
            lat        = np.asarray( data.lat[:] )
            lon        = np.asarray( data.lon[:] )
            print(f"lon.min(): {lon.min():.3f}, lon.max() = {lon.max():.3f}")
            lon[(180<=lon) & (lon<=360)] = lon[(180<=lon) & (lon<=360)] - 360
            print(f"lon.min(): {lon.min():.3f}, lon.max() = {lon.max():.3f}")
        return wind_speed, lat, lon, time_ascat


    def search_data_in_anomalies(self,mi,mlat,mlon,lat,lon,printQ=False,**kwargs):
        """
        Inputs:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of data locations. The key is the anomaly, or
        event id. Each value is a list of (data) indices for points that 
        were found within the mask.
        """
        anomalies = np.unique( mi[:,2], return_counts=True )
        locs = {}
        for i,anomaly in zip(range(len(anomalies[0])),anomalies[0]):
            if printQ: print(f"{i}, anomaly: {anomaly}")
            # Find mask indices for which the id is equal to anomaly:
            ix = np.argwhere( mi[:,2] == anomaly )
            # numpy.int32 to int:
            ano_int = int(anomaly)

            # Select mask points corresponding to this anomaly:
            mlat_ix = mlat[ mi[ix,0] ]
            mlon_ix = mlon[ mi[ix,1] ]

            if 'distance' in kwargs.keys():
                #d = kwargs['distance']
                locs[ano_int] = self.data_in_mask_box(mlat_ix,mlon_ix,lat,lon)
                #locs[ano_int] = self.data_in_mask(mlat_ix,mlon_ix,lat,lon,distance=d)
            else:
                locs[ano_int] = self.data_in_mask_box(mlat_ix,mlon_ix,lat,lon)
                #locs[ano_int] = self.data_in_mask(mlat_ix,mlon_ix,lat,lon)
        return locs


    def get_mask_data(self,mask_file,groupname):
        groupi = self.get_mask(mask_file,groupname)
        mi = np.asarray( groupi['mask_indices'][:] )
        mask_lat,mask_lon = self.get_navigation(mask_file)

        return mi,mask_lat,mask_lon


    def get_ASCAT_coordinates(self,filename,printQ=False):
        self.update_PODAAC_credentials(printQ)
        fs_s3 = s3fs.S3FileSystem(anon=False,
                                    key    = self.s3_creds['accessKeyId'], 
                                    secret = self.s3_creds['secretAccessKey'], 
                                    token  = self.s3_creds['sessionToken'])
        with fs_s3.open(filename,'rb') as f:
            if printQ: print(f"f = fs_s3.open({filename},'rb')")
            data = xr.open_dataset(f)
            lat  = np.asarray( data.lat[:] )
            lon  = np.asarray( data.lon[:] )
            if printQ: print(f"lon.min(): {lon.min()}, lon.max() = {lon.max()}")
            lon[(180<=lon) & (lon<=360)] = lon[(180<=lon) & (lon<=360)] - 360
            if printQ: print(f"lon.min(): {lon.min()}, lon.max() = {lon.max()}")
        return lat, lon


    def search_ascatb_files(self,ascatb_files,mi,mask_lat,mask_lon,printQ=False,**kwargs):
        """
        Input:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations. The key is
        the ascatb file name: locs[ascatb_file][anomaly_id] = list of
        data indices (those of mlat and mlon).

        Note: The Numpy function reshape is prefered to the flatten function
        to transform the `lat` and `lon` coordinates, to guarantee a known
        relationship with the reverse operation (2D==>1D). So, the 
        following equality is true:
        np.array_equal(lon,np.reshape(np.reshape(lon,lon.size),lon.shape))
        The reverse, 1D to 2D, transformation is:
        newlon2d = np.reshape(machin,lon.shape),
        which means the lon.shape must be known.
        """
        locs = {}
        for ascatb_file in ascatb_files:
            tStart = time.time()
            if printQ: print(f"ascatb_file: {ascatb_file}")
            lati, loni = self.get_ASCAT_coordinates(ascatb_file)
            # 2D ==> 1D:
            lat = np.reshape(lati,lati.size)
            lon = np.reshape(loni,loni.size)
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon,distance=d)
            else:
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon)
            locs[ascatb_file] = lo
            tEnd   = time.time()
            if printQ: print(f"{round(tEnd-tStart)} [s]")
        return locs


    @staticmethod
    def ASCAT_file_name_to_date(afilename):
        """
        ascat_YYYYMMDD_HHMMSS_SAT_ORBIT_SRV_T_SMPL(_CONT).l2_bufr
        Output:
            string 'YYYYMMDD_HHMMSS'
        """
        match = re.search("\d{8}_\d{6}",afilename)
        if match:
            st = match.group(0)
            return ''.join( st.split('_') )
        else:
            print("Error in ASCAT_file_name_to_date.")
            return None


    def get_data_indices(self,printQ=False,**kwargs):
        """
        Input:
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations,
        locs[ groupname[i] ][ ascatfile ][ anomaly ] = [data_index].
        The key `groupname[i]` is the mask time stamp (string) eg
        '202303010030'. The `ascatfile` key is the ASCAT file (string).
        It contains the path on the PODAAC server and the netcdf file
        name. The `anomaly` key is an integer (numpy.int32). 
        data indices (those of mlat and mlon). The values are a list of
        data points indices that are within a distance the mask data.
        """
        #--- step 1 ---:
        # Mask output hierarchy file:
        groupnames    = self.get_group_names()
        groupnames_dt = self.get_group_names_datetimes(groupnames)

        #--- step 2 ---:
        # Curation data dictionary:
        cf = "/data/code/data-dictionaries/tos2ca-data-collection-dictionary.json"
        cur = self.get_curation_dictionary(cf)
        ascatb_files,ascatb_datetimes = self.ascat_files_info('ASCATB-L2-25km',cur)

        mask_file = self.mask_file
        mask_lat,mask_lon = self.get_navigation(mask_file) # mask_lat.shape = (number,)

        locs = {}
        for i in range( len(groupnames)-1 ): # 96-1
            # debug:
            # if i>2:
            #     break

            sd = groupnames_dt[i]
            ed = groupnames_dt[i+1]
            ixs,ixe = self.search_files(sd,ed,ascatb_datetimes)
            ascatb_files_in_maski = [ascatb_files[ix] for ix in range(ixs,ixe+1)]

            if printQ:
                self.print_line('=',80)
                print(f"groupnames[{i}/{len(groupnames)-1}]: {groupnames[i]}")
                print(f"start_date (datetime): {sd}")
                print(f"end_date (datetime): {ed}")
                print(f"Look for ASCATB files between {ascatb_datetimes[0]}",end='')
                print(f" and {ascatb_datetimes[-1]}:")
                print(f"ixs = {ixs}, ixe = {ixe}")
                for afim in ascatb_files_in_maski:
                    print(afim)

            #--- step 3 ---:
            groupi = self.get_mask(mask_file,groupnames[i])
            mi = np.asarray( groupi['mask_indices'][:] )
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                locsi = self.search_ascatb_files(ascatb_files_in_maski,mi,mask_lat,mask_lon,distance=d)
            else:
                locsi = self.search_ascatb_files(ascatb_files_in_maski,mi,mask_lat,mask_lon)
            locs[groupnames[i]] = locsi
            if printQ: print('')

        return locs


    def ascatb_file2date(self,locs):
        """
        This function return a dictionary of all the ASCATB file names used
        in `locs`. The key is the full file name (with path) and the value
        the data file date only.
        This is necessary because the date is not sufficient (although
        necessary) to retrieve an ASCATB file. Other non unique fields
        exist.
        """
        # Set of (unique) ASCATB files:
        ascatb_files = { af for v1 in locs.values() for af in v1.keys() }
        return { af:self.ASCAT_file_name_to_date(af) for af in ascatb_files }


    def ascatb_date2file(self,locs):
        """
        This function return a dictionary of all the ASCATB file names used
        in `locs`. The key is the data file date only and the value, the
        full file name (with path).
        This is necessary because the date is not sufficient (although
        necessary) to retrieve an ASCATB file. Other non unique fields
        exist.
        """
        # Set of (unique) ASCATB files:
        ascatb_files = { af for v1 in locs.values() for af in v1.keys() }
        return { self.ASCAT_file_name_to_date(af):af for af in ascatb_files }


    def netcdf_variables_length(self,locs,printQ=False):
        """
        Calculate the length of each variable using the output the function
        get_data_indices():
            locs[ mask time stamp ][ ascatfile ][ anomaly ] = [data index]
            with
            mask time stamp (string): '202303010000'
            ascatfile (string): full path file name
            anomaly (integer): phenomenon index
        """
        data_length = {} # data_length[asctatbfile][anomaly] 
        for k1,v1 in locs.items():
            if printQ:
                self.print_line('=',80)
                print(f"mask time stamp: {k1}")
            for k2,v2 in v1.items():
                if printQ:
                    print(f"\tascat file name: {k2}")
                # ascat file date (string):
                ad = self.ASCAT_file_name_to_date(k2)
                for k3,v3 in v2.items():
                    # len(v3): data indices list length
                    if len(v3):
                        if printQ:
                            print(f"\t\tanomaly: {k3:2d}, len(v3) = {len(v3)}")
                        if ad not in data_length.keys():
                            data_length[ad] = {}
                            if printQ: print(f"\t\tdata_length[{ad}] = {{}}")
                        if k3 not in data_length[ad].keys():
                            data_length[ad][k3] = []
                            if printQ: print(f"\t\tdata_length[{ad}][{k3}] = []")
                        data_length[ad][k3].append( len(v3) )
                        if printQ:
                            print(f"\t\tdata_length[{ad}][{k3}].append( {len(v3)} )")
                            print(f"\t\tdata_length[{ad}][{k3}] = {data_length[ad][k3]}\n")
        return data_length


    def get_data(self,locs,printQ=False):
        """
        Gather data for netcdf variables
        data_length = {} # data_length[asctatbfile][anomaly] 
        locs[ groupname[i] ][ ascatfile ][ anomaly ] = [data index]
        """
        data = {}
        missing_value = 1.0E30

        for maski,v1 in locs.items():
            if printQ:
                self.print_line("=",80)
                print(f"maski: {maski}")

            for ascatb_file,v2 in v1.items():
                # ascat file date (string):
                ascatb_file_date = self.ASCAT_file_name_to_date(ascatb_file)
                wind_speed, lat, lon, time_ascat = self.get_ASCAT_data(ascatb_file)
                if printQ:
                    self.print_line2("=",72,pre='\t')
                    print(f"\tascatb_file: {ascatb_file}, ascatb_file_date: {ascatb_file_date}")
                    print(f"\twind_speed.shape = {wind_speed.shape}") # (1584, 42)

                for anomaly,v3 in v2.items():
                    anomalys = str(anomaly)
                    if printQ:
                        self.print_line2("=",64,pre='\t\t')
                        print(f"\t\tanomalys: {anomalys}")

                    data_ixs = locs[maski][ascatb_file][anomaly]
                    if len(data_ixs):
                        if printQ:
                            print(f"\t\t\t{anomalys} in {locs[maski][ascatb_file].keys()}: ",end='')
                            print(f"\n\t\t\t\t{data_ixs}")
                        if ascatb_file_date not in data.keys():
                            data[ascatb_file_date] = {}
                            if printQ: print(f"\t\t\tdata[{ascatb_file_date}] = {{}}")
                        if anomalys not in data[ascatb_file_date].keys():
                            data[ascatb_file_date][anomalys] = []
                            if printQ: print(f"data[{ascatb_file_date}][{anomalys}] = []")
                        self.print_line2("-",56,pre='\t\t\t')
                        for data_ix in data_ixs:
                            (i,j) = np.unravel_index(data_ix,wind_speed.shape)
                            if np.isnan(wind_speed[i,j]):
                                ws = missing_value
                            else:
                                ws = wind_speed[i,j]
                            if printQ:
                                print(f"\t\t\t{data_ix}: (wind_speed[{(i,j)}], lat[{(i,j)}], lon[{(i,j)}]) ")
                                print(f"\t\t\tdata[{ascatb_file_date}][{anomalys}].append({lat[i,j]},{lon[i,j]},{ws})")
                            data[ascatb_file_date][anomalys].append((lat[i,j],lon[i,j],ws))
        return data


    def create_netCDF_file(self,data,ncfilename,printQ=False):
        """
        """
        if printQ:
            print(f"Create netCDF file {ncfilename}")
        ncfile = nc.Dataset(ncfilename, "w", format="NETCDF4")

        ncfile.format = 'netCDF-4'
        ncfile.variable = "wind_speed"
        ncfile.dataset = "ASCATB-L2-25km"
        ncfile.units = "m s-1";
        ncfile.missing_value = "1.0E30"
        ncfile.website = 'https://tos2ca-dev1.jpl.nasa.gov'
        ncfile.project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
        ncfile.institution = 'NASA Jet Propulsion Laboratory'
        ncfile.production_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        ncgrp1 = {}
        ncgrp2 = {}
        for ascatb_date,v1 in data.items():
            if printQ:
                self.print_line('=',80)
                print(f"create group: ncgrp1[{ascatb_date}] = ncfile."
                    f"createGroup({ascatb_date})")
            ncgrp1[ascatb_date] = ncfile.createGroup(ascatb_date)
            ncgrp2[ascatb_date] = {}

            for anomalys,v2 in v1.items():
                print(f"\t\tanomaly: {anomalys}, len(v2) = {len(v2)}")
                if anomalys not in ncgrp2.get(ascatb_date).keys():
                    strgrp2 = "/" + ascatb_date + "/"
                    ncgrp2[ascatb_date][anomalys] = ncfile.createGroup(strgrp2+anomalys)
                    ncfile[ascatb_date][anomalys].createDimension('data_point',3)
                    nobs = len(v2)
                    if printQ:
                        print(f"\t\tcreate group: ncgrp2[{ascatb_date}][{anomalys}]"
                            f" = ncfile.createGroup({strgrp2+anomalys})")
                        print(f"\t\tncfile[{ascatb_date}][{anomalys}]."
                            f"createDimension('observation',{nobs})\n")
                    ncfile[ascatb_date][anomalys].createDimension('observation',nobs)
                    ws_var = ncfile[ascatb_date][anomalys].createVariable('wind_speed', 'f4', ('observation', 'data_point',))
                    ws_var.description = 'Each row is a pixel with the columns indicating (lat,lon,wind_speed).'
                    ws_var[:] = np.array(v2, dtype=np.float32)
                else:
                    print("Error in create_netCDF_file()")
        ncfile.close()


    @staticmethod
    def ASCAT_orbit(afilename):
        """
        ascat_YYYYMMDD_HHMMSS_SAT_ORBIT_SRV_T_SMPL(_CONT).l2_bufr
        Output:
            string 'ORBIT'. ORBIT is the orbit number (00000-99999)
        """
        match = re.search("_\d{5}_",afilename)
        if match:
            st = match.group(0)
            return st[1:-1]
        else:
            print("Error in ASCAT_orbit.")
            return None


    def get_close_orbit_files(self,ix,ascatb_files,printQ=False):
        # Get orbit, orbit-1 and orbit+1:
        orbit = int(self.ASCAT_orbit(ascatb_files[ix]))
        orbitm = orbit - 1
        orbitp = orbit + 1
        if printQ: print(f"orbitm = {orbitm}, orbitp = {orbitp}")
        # Get orbits corresponding to file[i-1], file[i], file[i+1]:
        orbitm2 = int(self.ASCAT_orbit(ascatb_files[ix-1]))
        orbitp2 = int(self.ASCAT_orbit(ascatb_files[ix+1]))
        if printQ: print(f"orbitm2 = {orbitm2}, orbitp2 = {orbitp2}")
        # Check (equality/consistency):
        if orbitm == orbitm2 and orbitp == orbitp2:
            return [ascatb_files[ix-1],ascatb_files[ix],ascatb_files[ix+1]]
        else:
            print("Error in get_close_orbit_files()")
            return None
        

    def search_close_ascatb_files(self,ascatb_files,mi,mask_lat,mask_lon,printQ=False,**kwargs):
        """
        Input:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations. The key is
        the ascatb file name: locs[ascatb_file][anomaly_id] = list of
        data indices (those of mlat and mlon).

        Note: The Numpy function reshape is prefered to the flatten function
        to transform the `lat` and `lon` coordinates, to guarantee a known
        relationship with the reverse operation (2D==>1D). So, the 
        following equality is true:
        np.array_equal(lon,np.reshape(np.reshape(lon,lon.size),lon.shape))
        The reverse, 1D to 2D, transformation is:
        newlon2d = np.reshape(machin,lon.shape),
        which means the lon.shape must be known.
        """
        locs = {}
        for ascatb_file in ascatb_files:
            tStart = time.time()
            if printQ: print(f"ascatb_file: {ascatb_file}")
            lati, loni = self.get_ASCAT_coordinates(ascatb_file)
            # 2D ==> 1D:
            lat = np.reshape(lati,lati.size)
            lon = np.reshape(loni,loni.size)
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon,distance=d)
            else:
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon)
            locs[ascatb_file] = lo
            tEnd   = time.time()
            if printQ: print(f"{round(tEnd-tStart)} [s]")
        return locs


    def search_ascatb_files_3t(self,ascatb_files,mi,mask_lat,mask_lon,printQ=False,**kwargs):
        """
        Input:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations. The key is
        the ascatb file name: locs[ascatb_file][anomaly_id] = list of
        data indices (those of mlat and mlon).

        Note: The Numpy function reshape is prefered to the flatten function
        to transform the `lat` and `lon` coordinates, to guarantee a known
        relationship with the reverse operation (2D==>1D). So, the 
        following equality is true:
        np.array_equal(lon,np.reshape(np.reshape(lon,lon.size),lon.shape))
        The reverse, 1D to 2D, transformation is:
        newlon2d = np.reshape(machin,lon.shape),
        which means the lon.shape must be known.
        """
        locs = {}
        for ascatb_file,close_files in ascatb_files.items():
            if printQ: print(f"ascatb_file: {ascatb_file}")
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                lo = self.search_close_ascatb_files(close_files,mi,mask_lat,mask_lon,distance=d)
            else:
                lo = self.search_close_ascatb_files(close_files,mi,mask_lat,mask_lon)
            locs[ascatb_file] = lo
        return locs


    def get_data_indices_3t(self,varname,printQ=False,**kwargs):
        """
        Input:
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations,
        locs[ groupname[i] ][ ascatfile ][ ascatfile_t ][ anomaly ] = [data_index].
        The key `groupname[i]` is the mask time stamp (string) eg
        '202303010030'. The `ascatfile` key is the ASCAT file (string).
        It contains the path on the PODAAC server and the netcdf file
        name. The `anomaly` key is an integer (numpy.int32). 
        data indices (those of mlat and mlon). The values are a list of
        data points indices that are within a distance the mask data.
        """
        #--- step 1 ---:
        # Mask output hierarchy file:
        groupnames    = self.get_group_names()
        groupnames_dt = self.get_group_names_datetimes(groupnames)
        # groupnames_dt_pd = pd.Series( groupnames_dt )

        #--- step 2 ---:
        # Curation data dictionary:
        cf = "/data/code/data-dictionaries/tos2ca-data-collection-dictionary.json"
        cur = self.get_curation_dictionary(cf)
        if varname == 'ASCATB-L2-25km' or varname == 'ASCATC-L2-25km':
            ascat_files,ascat_datetimes = self.ascat_files_info(varname,cur)
        else:
            print("Error in Curation.get_data_indices_3t() varname: ",end='')
            print(f"""{varname}. It should be 'ASCATB-L2-25km' or""",end='')
            print(""" 'ASCATC-L2-25km'.""")

        mask_lat,mask_lon = self.get_navigation(self.mask_file) # mask_lat.shape = (number,)

        locs = {}
        for i in range( len(groupnames)-1 ): # 96-1
            # debug:
            # print("debug")
            # if i>1:
            #     break

            sd = groupnames_dt[i]
            ed = groupnames_dt[i+1]
            ixs,ixe = self.search_files(sd,ed,ascat_datetimes)
            d = {}
            ascat_files_in_maski = []
            for ix in range(ixs,ixe+1):
                d[ascat_files[ix]] = self.get_close_orbit_files(ix,ascat_files)
                ascat_files_in_maski.append(ascat_files[ix])

            if printQ:
                self.print_line('=',80)
                print(f"groupnames[{i}/{len(groupnames)-1}]: {groupnames[i]}")
                print(f"start_date (datetime): {sd}")
                print(f"end_date (datetime): {ed}")
                print(f"Look for ASCATB files between {ascat_datetimes[0]}",end='')
                print(f" and {ascat_datetimes[-1]}:")
                print(f"ixs = {ixs}, ixe = {ixe}")
                for afim in ascat_files_in_maski:
                    print(afim)

            #--- step 3 ---:
            groupi = self.get_mask(self.mask_file,groupnames[i])
            mi = np.asarray( groupi['mask_indices'][:] )
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                locsi = self.search_ascat_files_3t(d,mi,mask_lat,mask_lon,distance=d)
            else:
                locsi = self.search_ascat_files_3t(d,mi,mask_lat,mask_lon)
            locs[groupnames[i]] = locsi
            if printQ: print('')

        return locs


    def get_data_3t(self,locs,printQ=False):
        """
        Gather data for netcdf variables
        Input:
        locs[ groupname[i] ][ ascatfile ][ ascatfile_t ][ anomaly ] = [data_index]
        Ouput:
        data[ascatb_file_date][ascatb_file_date_t][anomalys] = list( (lat[i,j],lon[i,j],ws) )
                                                = list( (numpy.float32, numpy.float32, float) )
        """
        data = {}
        missing_value = 1.0E30

        for i,(maski,v1) in enumerate(locs.items()):
            if printQ:
                self.print_line("=",80)
                print(f"maski: {maski}, {i}/{len(locs)-1}")

            for ascatb_file,v2 in v1.items():
                # ascat file date (string):
                ascatb_file_date = self.ASCAT_file_name_to_date(ascatb_file)
                if printQ:
                    self.print_line2("=",72,pre='\t')
                    print(f"\tascatb_file: {ascatb_file}, ascatb_file_date: {ascatb_file_date}")

                # Loops through {t-1,t,t+1} ascat files:
                for k,(ascatb_file_t,v3) in enumerate(v2.items()): # k in {0,1,2}
                    # ascat file date (string):
                    ascatb_file_date_t = self.ASCAT_file_name_to_date(ascatb_file_t)
                    wind_speed, lat, lon, time_ascat = self.get_ASCAT_data(ascatb_file_t)
                    if printQ:
                        self.print_line2("=",72,pre='\t\t')
                        print(f"\t\tk = {k}/{len(v2)-1}")
                        print(f"\t\tascatb_file_t: {ascatb_file_t}")
                        print(f"\t\tascatb_file_date_t: {ascatb_file_date_t}")
                        print(f"\t\twind_speed.shape = {wind_speed.shape}") # (1584, 42)

                    for anomaly,v4 in v3.items():
                        anomalys = str(anomaly)
                        if printQ:
                            self.print_line2("=",64,pre='\t\t\t')
                            print(f"\t\t\tanomalys: {anomalys}")

                        data_ixs = locs[maski][ascatb_file][ascatb_file_t][anomaly]
                        if len(data_ixs):
                            if printQ:
                                print(f"\t\t\t\t{anomalys} in {locs[maski][ascatb_file].keys()}: ",end='')
                                print(f"\n\t\t\t\t\t{data_ixs}")
                            if ascatb_file_date not in data.keys():
                                data[ascatb_file_date] = {}
                                if printQ: print(f"\t\t\t\tdata[{ascatb_file_date}] = {{}}")
                            if ascatb_file_date_t not in data[ascatb_file_date].keys():
                                data[ascatb_file_date][ascatb_file_date_t] = {}
                                if printQ: print(f"\t\t\t\tdata[{ascatb_file_date}][{ascatb_file_date_t}] = {{}}")
                            if anomalys not in data[ascatb_file_date][ascatb_file_date_t].keys():
                                data[ascatb_file_date][ascatb_file_date_t][anomalys] = []
                                if printQ: print(f"\t\t\t\tdata[{ascatb_file_date}][{ascatb_file_date_t}][{anomalys}] = []")
                            self.print_line2("-",56,pre='\t\t\t\t')
                            for data_ix in data_ixs:
                                (i,j) = np.unravel_index(data_ix,wind_speed.shape)
                                if np.isnan(wind_speed[i,j]):
                                    ws = missing_value
                                else:
                                    ws = wind_speed[i,j]
                                if printQ:
                                    print(f"\t\t\t\t{data_ix}: (wind_speed[{(i,j)}], lat[{(i,j)}], lon[{(i,j)}]) ")
                                    print(f"\t\t\t\tdata[{ascatb_file_date}][{ascatb_file_date_t}][{anomalys}].append({lat[i,j]},{lon[i,j]},{ws})")
                                data[ascatb_file_date][ascatb_file_date_t][anomalys].append((lat[i,j],lon[i,j],ws))

                if ascatb_file_date in data.keys():
                    add_var = False
                    for k,ascatb_file_t in enumerate(v2.keys()): # k in {0,1,2}
                        ascatb_file_date_t = self.ASCAT_file_name_to_date(ascatb_file_t)
                        if ascatb_file_date_t in data[ascatb_file_date].keys():
                            add_var = True
                            break;
                    if add_var:
                        for k,ascatb_file_t in enumerate(v2.keys()): # k in {0,1,2}
                            ascatb_file_date_t = self.ASCAT_file_name_to_date(ascatb_file_t)
                            if ascatb_file_date_t not in data[ascatb_file_date].keys():
                                data[ascatb_file_date][ascatb_file_date_t] = {}
                                if printQ:
                                    print(f"\t\t\t\tdata[{ascatb_file_date}][{ascatb_file_date_t}] = {{}}, k = {k}")
                            # else: # debug
                            #     print(f"\t\t\t\tdata[{ascatb_file_date}][{ascatb_file_date_t}] exists, k = {k}")
        return data


    @staticmethod
    def serialize_data3t(data3t,printQ=False):
        """
        Input:
            data[ascatb_file_date][ascatb_file_date_t][anomalys] =
                list( (lat[i,j],lon[i,j],ws) ) =
                list( (numpy.float32, numpy.float32, float) )
        Output:
            data[ascatb_file_date][ascatb_file_date_t][anomalys] =
                list( (lat[i,j],lon[i,j],ws) ) =
                list( (float, float, float) )
        """
        # right now data is a list of tuple (numpy.float32, numpy.float32, float):
        data = {}
        for ascatb_file_date,v1 in data3t.items():
            if printQ: print(f"{ascatb_file_date}")
            if ascatb_file_date not in data.keys():
                data[ascatb_file_date] = {}
            for ascatb_file_date_t,v2 in v1.items():
                if printQ: print(f"\t{ascatb_file_date_t}")
                if ascatb_file_date_t not in data[ascatb_file_date].keys():
                    data[ascatb_file_date][ascatb_file_date_t] = {}
                for anomalys,v3 in v2.items():
                    if anomalys not in data[ascatb_file_date][ascatb_file_date_t].keys():
                        data[ascatb_file_date][ascatb_file_date_t][anomalys] = []
                    print(f"\t\t{anomalys}: {v3}")
                    for vi in v3:
                        t = ( float(vi[0]), float(vi[1]), float(vi[2]) )
                        data[ascatb_file_date][ascatb_file_date_t][anomalys].append(t)
        return data


    def get_data_3t2(self,locs,printQ=False):
        """
        Gather data for netcdf variables
        Input:
        locs[ groupname[i] ][ ascatfile ][ ascatfile_t ][ anomaly ] = [data_index]
        Ouput:
        data[ascat_file_date][anomalys] = list( (lat[i,j],lon[i,j],ws) )
                                         = list( (numpy.float32, numpy.float32, float) )
        """
        data = {}
        missing_value = 1.0E30

        for i,(maski,v1) in enumerate(locs.items()):
            if printQ:
                self.print_line("=",80)
                print(f"maski: {maski}, {i}/{len(locs)-1}")

            for ascat_file,v2 in v1.items():
                # ascat file date (string):
                ascat_file_date = self.ASCAT_file_name_to_date(ascat_file)
                if printQ:
                    self.print_line2("=",72,pre='\t')
                    print(f"\tascat_file: {ascat_file}, ascat_file_date: {ascat_file_date}")

                # {full ascat file (faf): '20230228231800', ...}:
                d1 = { s:self.ASCAT_file_name_to_date(s) for s in v2.keys() }
                # date string to datetime dictionary:
                # {faf:('20230228231800',datetime.datetime(2023, 2, 28, 23, 18)), ...}
                d2 = { s:(d1[s],self.anomaly_datetime(d1[s])) for s in v2.keys() }
                # Sort dictionary by values (datetimes):
                od = OrderedDict( sorted(d2.items(),key=lambda x: x[1][1]) )
                # ordered date strings:
                ods = [ v[0] for v in d2.values() ]

                # Loops through {t-1,t,t+1} ascat files (in order):
                for k,(ascat_file_t,(ascat_file_date_t,ascat_file_datetime_t)) in enumerate(od.items()):
                    # for ascat_file_date_t,v2 in v1.items():
                    print(f"\t{k}: {ascat_file_date_t}, {ascat_file_datetime_t}")
                    v3 = v2.get(ascat_file_t)
                    wind_speed, lat, lon, time_ascat = self.get_ASCAT_data(ascat_file_t)
                    if printQ:
                        self.print_line2("=",72,pre='\t\t')
                        print(f"\t\tk = {k}/{len(v2)-1}")
                        print(f"\t\tascat_file_t: {ascat_file_t}")
                        print(f"\t\tascat_file_date_t: {ascat_file_date_t}")
                        print(f"\t\twind_speed.shape = {wind_speed.shape}") # (1584, 42)

                    for anomaly,v4 in v3.items():
                        anomalys = str(anomaly)
                        if printQ:
                            self.print_line2("=",64,pre='\t\t\t')
                            print(f"\t\t\tanomalys: {anomalys}")

                        data_ixs = v4
                        if len(data_ixs):
                            if printQ:
                                print(f"\t\t\tlocs[{maski}]")
                                print(f"\t\t\t[{ascat_file}]")
                                print(f"\t\t\t[{ascat_file_t}]")
                                print(f"\t\t\t[{anomaly}]: ")
                                print(f"\t\t\t\t{data_ixs}")
                            if ascat_file_date not in data.keys():
                                data[ascat_file_date] = {}
                                if printQ: print(f"\t\t\tdata[{ascat_file_date}] = {{}}")
                            if anomalys not in data[ascat_file_date].keys():
                                data[ascat_file_date][anomalys] = (ods,[[],[],[]])
                                if printQ: print(f"\t\t\tdata[{ascat_file_date}][{anomalys}] = ({ods},[[],[],[]])")
                            self.print_line2("-",56,pre='\t\t\t\t')
                            if len(data_ixs):
                                for data_ix in data_ixs:
                                    (i,j) = np.unravel_index(data_ix,wind_speed.shape)
                                    if np.isnan(wind_speed[i,j]):
                                        ws = missing_value
                                    else:
                                        ws = wind_speed[i,j]
                                    if printQ:
                                        print(f"\t\t\t\t{data_ix}: (wind_speed[{(i,j)}], lat[{(i,j)}], lon[{(i,j)}]) ")
                                        print(f"\t\t\t\tdata[{ascat_file_date}][{anomalys}][1][{k}].append(({lat[i,j]},{lon[i,j]},{ws}))")
                                    data[ascat_file_date][anomalys][1][k].append((float(lat[i,j]),float(lon[i,j]),float(ws)))

        return data


    @staticmethod
    def fill_data3t2(data3t,printQ=False):
        """
        Fill data to get a regular array.
        Input:
            data[ascatb_file_date][anomalys] =
            (
            ['20230228231800', '20230301005700', '20230228213600'],
            [ (lat[i,j],lon[i,j],ws), (lat[i,j],lon[i,j],ws), (lat[i,j],lon[i,j],ws) ]
            )
            with list( (lat[i,j],lon[i,j],ws) ) =
                 list( (numpy.float32, numpy.float32, float) )
        """
        # right now data is a list of tuple (numpy.float32, numpy.float32, float):
        # tuple of None:
        _FillValue = 1.0E30
        t = ( _FillValue, _FillValue, _FillValue )
        data = {}
        for ascatb_file_date,v1 in data3t.items():
            if printQ: print(f"{ascatb_file_date}")
            if ascatb_file_date not in data.keys():
                data[ascatb_file_date] = {}
            for anomalys,v2 in v1.items():
                if printQ: print(f"\t{anomalys}")
                if anomalys not in data[ascatb_file_date].keys():
                    data[ascatb_file_date][anomalys] = (v2[0],[[],[],[]])
                maxLength = -1
                for k,listk in enumerate(v2[1]):
                    print(f"\t\t{v2[0][k]}, len(list[{k}]): {len(listk)}")
                    if len(listk) > maxLength:
                        maxLength = len(listk)
                for k,listk in enumerate(v2[1]):
                    for vi in listk:
                        data[ascatb_file_date][anomalys][1][k].append(vi)
                    for i in range(len(listk),maxLength):
                        data[ascatb_file_date][anomalys][1][k].append(t)
        return data
    

    def create_netCDF_file3t(self,data,jobInfo,ncfilename,printQ=False):
        """
        Input:
        data[ascatb_file_date][anomalys] =
            (
            ['20230228231800', '20230301005700', '20230228213600'],
            [ (lat[i,j],lon[i,j],ws), (lat[i,j],lon[i,j],ws), (lat[i,j],lon[i,j],ws) ]
            )
            with list( (lat[i,j],lon[i,j],ws) ) =
                 list( (float, float, float) )
        """
        info = self.load_json(self.data_col_dict)
        variable    = jobInfo['variable']
        dataset     = jobInfo['dataset']
        phdefJobID  = jobInfo['phdefJobID']
        jobID       = jobInfo['jobID']
        units       = info[dataset]['units'][variable]
        productInfo = info[dataset]['productInfo']
        fullName    = info[dataset]['fullName']

        if printQ:
            print(f"Create netCDF file {ncfilename}")
        ncfile = nc.Dataset(ncfilename, "w", format="NETCDF4")
        ncfile.format = 'netCDF-4'
        ncfile.Variable = variable # "wind_speed"
        ncfile.Dataset = dataset # "ASCATB-L2-25km"
        ncfile.Units = units # "m s-1";
        ncfile.MissingValue = "1.0E30"
        ncfile.References = 'https://tos2ca-dev1.jpl.nasa.gov'
        ncfile.Project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
        ncfile.Institution = 'NASA Jet Propulsion Laboratory'
        ncfile.ProductionTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ncfile.PhDefJobID = jobInfo['phdefJobID']
        ncfile.ProductInfo = productInfo
        ncfile.FullName = fullName
        ncfile.SpatialCoverage = 'global'
        ncfile.FileFormat = 'NetCDF-4/HDF-5'
        ncfile.DataResolution = '50km (effective)'

        ncfile.createGroup("navigation")
        if printQ:
            self.print_line('=',80)
            print("""create group: ncfile.createGroup("navigation")""")
        mask_file = self.mask_file
        mask_lat,mask_lon = self.get_navigation(mask_file)
        latDim = ncfile['navigation'].createDimension('lat', len(mask_lat))
        lonDim = ncfile['navigation'].createDimension('lon', len(mask_lon))
        lat = ncfile['navigation'].createVariable('lat', 'f4', ('lat',), zlib=True, complevel=9)
        lon = ncfile['navigation'].createVariable('lon', 'f4', ('lon',), zlib=True, complevel=9)
        lat[:] = mask_lat
        lon[:] = mask_lon
        s1 = 'The lat and lon are provided here to reconstruct the original global grid fr'
        s2 = 'om the dataset that created the masks.  They represent the centroid of the g'
        s3 = 'rid cell.'
        ncfile['navigation'].Description = s1+s2+s3

        curationHierarchy = {}
        observation_dims = ['observation_tm','observation_t','observation_tp']
        for ascatb_date,v1 in data.items():
            if printQ:
                self.print_line('=',80)
                print(f"create group: ncfile.createGroup({ascatb_date})")
                curationHierarchy[ascatb_date] = {}
            ncfile.createGroup(ascatb_date)
            # ncgrp1[ascat_date] = f"ncfile.createGroup({ascat_date})" # debug

            for anomalys,v2 in v1.items():
                if printQ: print(f"\t\tanomaly: {anomalys}, {v2[0]}, {len(v2[1])}, {len(v2[1][0])}")
                curationHierarchy[ascatb_date][anomalys] = [variable]
                if anomalys not in ncfile[ascatb_date].groups.keys():
                    strgrp2 = "/" + ascatb_date + "/"
                    ncfile.createGroup(strgrp2+anomalys)
                    ncfile[ascatb_date][anomalys].createDimension('time_step',3)
                    ncfile[ascatb_date][anomalys].createDimension('observation',len(v2[1][0]))
                    ncfile[ascatb_date][anomalys].createDimension('data_point',3)
                    if printQ:
                        print(f"\t\tcreate group: ncfile.createGroup({strgrp2+anomalys})")
                        print(f"\t\tncfile[{ascatb_date}][{anomalys}].createDimension('time_step',3)")
                        print(f"\t\tncfile[{ascatb_date}][{anomalys}].createDimension('observation',{len(v2[1][0])})")
                        print(f"\t\tncfile[{ascatb_date}][{anomalys}].createDimension('data_point',3)")
                    ws_var = ncfile[ascatb_date][anomalys].createVariable('wind_speed', 'f4',
                                                            ('time_step','observation', 'data_point',))
                    ws_var.description = 'Each row is a pixel with the columns indicating (lat,lon,wind_speed).'                   
                    ws_var.Times = ','.join(v2[0])
                    ws_var.TimeIndexing = 'index 0 = t-1; index 1 = t; index 2 = t+1'
                    ws_var.Units = "m s-1"
                    ws_var[:] = np.asarray(v2[1], dtype=np.float32)
                    
        ncfile.close()
        getCurationHierarchy(jobID, curationHierarchy)



    def search_ascat_files_3t(self,ascatb_files,mi,mask_lat,mask_lon,printQ=False,**kwargs):
        """
        Input:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations. The key is
        the ascatb file name: locs[ascatb_file][anomaly_id] = list of
        data indices (those of mlat and mlon).

        Note: The Numpy function reshape is prefered to the flatten function
        to transform the `lat` and `lon` coordinates, to guarantee a known
        relationship with the reverse operation (2D==>1D). So, the 
        following equality is true:
        np.array_equal(lon,np.reshape(np.reshape(lon,lon.size),lon.shape))
        The reverse, 1D to 2D, transformation is:
        newlon2d = np.reshape(machin,lon.shape),
        which means the lon.shape must be known.
        """
        locs = {}
        for ascatb_file,close_files in ascatb_files.items():
            if printQ: print(f"ascatb_file: {ascatb_file}")
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                lo = self.search_close_ascat_files(close_files,mi,mask_lat,mask_lon,distance=d)
            else:
                lo = self.search_close_ascat_files(close_files,mi,mask_lat,mask_lon)
            locs[ascatb_file] = lo
        return locs
    

    def search_close_ascat_files(self,ascatb_files,mi,mask_lat,mask_lon,printQ=False,**kwargs):
        """
        Input:
            mi: group[i] mask
                groupi = c.get_mask(mask_file,groupnamei)
                mi = np.asarray( groupi['mask_indices'][:] )
            mlat: mask latitude,  type: numpy.ndarray, shape: (length,1)
            mlon: mask longitude, type: numpy.ndarray, shape: (length,1)
            lat: data latitude,   type: numpy.ndarray, shape: (length',1)
            lon: data latitude,   type: numpy.ndarray, shape: (length',1)
            distance (optional): distance in degrees
        Ouput:
        locs: dictionnary of dictionnaries of data locations. The key is
        the ascatb file name: locs[ascatb_file][anomaly_id] = list of
        data indices (those of mlat and mlon).

        Note: The Numpy function reshape is prefered to the flatten function
        to transform the `lat` and `lon` coordinates, to guarantee a known
        relationship with the reverse operation (2D==>1D). So, the 
        following equality is true:
        np.array_equal(lon,np.reshape(np.reshape(lon,lon.size),lon.shape))
        The reverse, 1D to 2D, transformation is:
        newlon2d = np.reshape(machin,lon.shape),
        which means the lon.shape must be known.
        """
        locs = {}
        for ascatb_file in ascatb_files:
            tStart = time.time()
            if printQ: print(f"ascatb_file: {ascatb_file}")
            lati, loni = self.get_ASCAT_coordinates(ascatb_file)
            # 2D ==> 1D:
            lat = np.reshape(lati,lati.size)
            lon = np.reshape(loni,loni.size)
            if 'distance' in kwargs.keys():
                d = kwargs['distance']
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon,distance=d)
            else:
                lo = self.search_data_in_anomalies(mi,mask_lat,mask_lon,lat,lon)
            locs[ascatb_file] = lo
            tEnd   = time.time()
            if printQ: print(f"{round(tEnd-tStart)} [s]")
        return locs
    

    def print_indices_locations(self,locs3t):
        for k1,v1 in locs3t.items():
            self.print_line('=',80)
            print(f"mask time stamp: {k1}")
            for k2,v2 in v1.items():
                print(f"\tascat file name: {k2}")
                for k3,v3 in v2.items():
                    print(f"\tascat file t name: {k3}")
                    for k4,v4 in v3.items():
                        # len(v4): data indices list length
                        if len(v4):
                            print(f"\t\tanomaly: {k4}, len(v4) = {len(v4)}")


    def print_data3t2(self,data3t2):
        for ascat_file_date,v1 in data3t2.items():
            print(f"{ascat_file_date}")
            for anomalys,v2 in v1.items():
                print(f"\t{anomalys}")
                for k,listk in enumerate(v2[1]):
                    print(f"\t\t{v2[0][k]}, len(listk) = {len(listk)}")
    
    def get_curation_information(self,curjobID):
        """
        curjobID: curation job ID (integer)
        This file used parts of the function merra2_curator from
        anomaly-detection/src/iolib/merra2.py
        """
        db, cur = openDB()
        updateStatus(db, cur, curjobID, 'running')
        jobInfo = getJobInfo(cur, curjobID)[0]
        phdefJobInfo = getJobInfo(cur, jobInfo['phdefJobID'])[0]

        startDate = phdefJobInfo['startDate']
        endDate = phdefJobInfo['endDate']
        phdefJobID = jobInfo['phdefJobID']

        print(f"startDate: {startDate}")
        print(f"endDate: {endDate}")
        print(f"phdefJobID: {phdefJobID}")

        sql = f'SELECT location, type FROM output WHERE jobID={phdefJobID} AND type IN ("masks", "toc", "hierarchy")'
        cur.execute(sql)
        results = cur.fetchall()
        for result in results:
            if result['type'] == 'masks':
                self.mask_file = result['location'] # used line
                print(f"mask_file: {self.mask_file}")
            if result['type'] == 'toc':
                tocFile = result['location']
                print(f"tocFile: {tocFile}")
            if result['type'] == 'hierarchy':
                self.hierarchyFile = result['location'] # used line 249
                print(f"hierarchyFile: {self.hierarchyFile}")

        closeDB(db)
        jobInfo['jobID'] = curjobID
        jobInfo['startDate'] = startDate
        
        return jobInfo, phdefJobInfo
    

    def create_ASCAT_curation_file(self,curjobID,printQ=False):
        """
        curjobID: curation job ID (integer)
        """
        jobInfo,_ = self.get_curation_information(curjobID)
        dataset = jobInfo['dataset']
        print(f"phenomenon definition job ID: {jobInfo['phdefJobID']}")
        print(f"dataset: {dataset}")
        
        if dataset != 'ASCATB-L2-25km' and dataset != 'ASCATC-L2-25km':
            print("Error in Curation.create_ASCAT_curation_file(): "
                  "dataset should be either ASCATB-L2-25km or ASCATC-L2-25km.")
            return None
        locs      = self.get_data_indices_3t(dataset,printQ) # 127 m
        data3t2   = self.get_data_3t2(locs,printQ)
        data_full = self.fill_data3t2(data3t2,printQ)
        ncfilename = '/data/tmp/%s-Curated-Data.nc4' % curjobID
        self.create_netCDF_file3t(data_full,jobInfo,ncfilename,printQ,)
        
        db, cur = openDB()
        
        uploadInfo = {}
        uploadInfo['filename'] = ncfilename
        uploadInfo['type'] = 'curated subset'
        uploadInfo['startDateTime'] = jobInfo['startDate']
        s3Upload(curjobID, uploadInfo, 'tos2ca-dev1', db, cur)

        uploadInfo = {}
        uploadInfo['filename'] = '/data/tmp/%s-Curation-Hierarchy.json' % curjobID
        uploadInfo['type'] = 'hierarchy'
        uploadInfo['startDateTime'] = jobInfo['startDate']
        s3Upload(curjobID, uploadInfo, 'tos2ca-dev1', db, cur)
        closeDB(db)
        return locs,data3t2,data_full
    
def ascat_curator(jobID):
    """
    :param jobID: curation jobID
    :type jobID: int
    """
    c = Curation()
    c.create_ASCAT_curation_file(jobID,printQ=True)