import numpy as np
import json
import netCDF4 as nc
import os
import shapely
from pyproj import Transformer
from shapely import wkt 
from shapely.geometry import Point, shape
from datetime import datetime

def calfire_reader(calfireGJ):
    """
    This function takes CA wildfire footprint information from
    the Cal Fire website and creates a netCDF-4 mask file for every 
    wildfire.  Each mask file will only have 1 timestamp, which 
    will be the containment time of the fire.
    Unlike the other readers, this does not read data out of S3, and
    must be passed a GeoJSON file with the mask data.
    :param calfireGJ: filename of the Cal Fire GeoJSON footprint file
    :type calfireGJ: str
    """
    filename = calfireGJ
    with open(filename) as f:
        gj = json.load(f)

    # box CA west coast
    print('Creating CA polygon')
    lons = np.arange(-125.20, -114.10, 0.005)
    lats = np.arange(32.30, 42.10, 0.005)

    numFeatures = len(gj['features'])
    for i in range(0, numFeatures):
        features = gj['features'][i]['geometry']
        properties = gj['features'][i]['properties']
        
        print(features['type'])
        print('Working on fire: %s - %s' % (properties['FIRE_NAME'], properties['OBJECTID']))

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326")

        if features['type'] == 'MultiPolygon':
            for i, v in enumerate(features['coordinates']):
                for j, w in enumerate(v):
                    for k, z in enumerate(w):
                        info = transformer.transform(z[0],z[1])
                        features['coordinates'][i][j][k] = [info[1], info[0]]
        elif features['type'] == 'Polygon':
            for i, v in enumerate(features['coordinates']):
                for j, w in enumerate(v):
                    info = transformer.transform(w[0],w[1])
                    features['coordinates'][i][j] = [info[1], info[0]]

        geo = shape(features)
        wktString = geo.wkt
        poly = wkt.loads(wktString)

        print('Cheking intersections')
        info = []
        found = 0
        for i, thisLat in enumerate(lats):
            for j, thisLon in enumerate(lons):
                point = Point(thisLon, thisLat)
                try:
                    if poly.contains(point):
                        info.append([i, j, 1])
                        print('found')
                        found = 1
                    #else:
                    #    info.append([thisLat, thisLon, 0])
                except shapely.errors.GEOSException:
                    print('Bad point found.')
                    info.append([thisLat, thisLon, 0])

        if found:
            # Setup netCDF-4 file
            if properties['FIRE_NAME']  == None:
                properties['FIRE_NAME'] = 'None'
            ncFilename = '/data/tmp/' + str(properties['OBJECTID']) + '-' + properties['FIRE_NAME'].replace('/', '+').replace('\\', '+') + '-Mask-Output.nc4'
            print('Creating: %s' % ncFilename)
            ncFile = nc.Dataset(ncFilename, 'w', format='NETCDF4')
            ncFile.Project = 'Thematic Observation Search, Segmentation, Collation and Analysis (TOS2CA)'
            ncFile.Institution = 'NASA Jet Propulsion Laboratory'
            ncFile.ProductionTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ncFile.SpatialCoverage = 'North America / West Coast'
            ncFile.FileFormat = 'NetCDF-4/HDF-5'
            ncFile.DataResolution = '0.005 x 0.005'
            ncFile.InputDataset = 'California Fire Perimeters (all)'
            ncFile.InputSource = 'https://gis.data.ca.gov/datasets/CALFIRE-Forestry::california-fire-perimeters-all-1/'
            ncFile.InputFilename = 'California_Fire_Perimeters_7897608464518632307.geojson'
            try:
                ncFile.ALARM_DATE = properties['ALARM_DATE']
            except TypeError:
                ncFile.ALARM_DATE = 'None'
            try:
                ncFile.CONT_DATE = properties['CONT_DATE']
            except TypeError:
                ncFile.CONT_DATE = 'None'
            ncFile.STATE = properties['STATE']
            ncFile.AGENCY = properties['AGENCY']
            ncFile.FIRE_NAME = properties['FIRE_NAME']
            ncFile.OBJECTID = properties['OBJECTID']
            ncFile.FIRE_NUM = str(properties['FIRE_NUM'])

            # Write to netCDF-4 file
            # Navigation group 
            navGroup = ncFile.createGroup('/navigation')
            latDim = ncFile['navigation'].createDimension('lat', len(lats))
            lonDim = ncFile['navigation'].createDimension('lon', len(lons))
            lat = ncFile['navigation'].createVariable('lat', 'f4', ('lat',), zlib=True, complevel=9)
            lon = ncFile['navigation'].createVariable('lon', 'f4', ('lon',), zlib=True, complevel=9)
            lat[:] = np.asarray(lats, dtype=np.float32)
            lon[:] = np.asarray(lons, dtype=np.float32)

            # Mask/timestamp group
            maskGroup = ncFile.createGroup('/masks')
            try:
                containmentDate = properties['CONT_DATE']
                containmentDate = datetime.strptime(containmentDate[:-7], '%a, %d %b %Y %H:%M').strftime('%Y%m%d%H%M')
            except TypeError:
                containmentDate = '9999999999'
            timestampGroup = ncFile['masks'].createGroup(containmentDate)

            # Set dimensions and variable
            pixelDim = ncFile['masks'][containmentDate].createDimension('num_pixels', len(info))
            colDim = ncFile['masks'][containmentDate].createDimension('num_cols', 3)
            data = ncFile['masks'][containmentDate].createVariable('mask_indices', 'i4', ('num_pixels', 'num_cols',), zlib=True, complevel=9)
            data[:] = np.asarray(info, dtype=np.float32)
            data.Description = 'Each row is a pixel with the columns indicating (i, j, event_id).'

            # Close the netCDF-4 file
            ncFile.close()

            #Create JSON TOC File
            tocFilename = '/data/tmp/' + str(properties['OBJECTID']) + '-' + properties['FIRE_NAME'].replace('/', '+').replace('\\', '+') + '-TOC.json'
            try:
                alarmDate = properties['ALARM_DATE']
            except TypeError:
                alarmDate = None
            try:
                contDate = properties['CONT_DATE']
            except TypeError:
                contDate = None
            tocInfo = {"name": "Anomaly 1", "start_date": alarmDate, "endDate": contDate}
            tocJSON = json.dumps(tocInfo)
            with open(tocFilename, 'w') as f:
                f.write("[" + tocJSON + "]")

            #Create JSON hierarchy file
            hierarchyFilename = '/data/tmp/' + str(properties['OBJECTID']) + '-' + properties['FIRE_NAME'].replace('/', '+').replace('\\', '+') + '-Mask-Output-Hierarchy.json'
            hierarchyInfo = {"navigation": ["lat", "lon"], "masks": {containmentDate: ["mask_indices"]}}
            hierarchyJSON = json.dumps(hierarchyInfo)
            with open(hierarchyFilename, 'w') as f:
                f.write(hierarchyJSON)

        else:
            print('No intersections found.  Not producing file.')

    return