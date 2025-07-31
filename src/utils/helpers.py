import json
import netCDF4 as nc
import s3fs
import pandas as pd
import shapely.wkt as wkt
import numpy as np

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from operators.inequalities import *


def getOperatorClass(className):
    """
    Function that retuns a class based on the name of the operator requested
    :param className: name of requested inequality
    :type className: string
    :return thisClass: class of the inequality
    :type thisClass: class 
    """
    inequalities = {
        'lessThan': lessThan,
        'lessThanOrEqualTo': lessThanOrEqualTo,
        'greaterThan': greaterThan,
        'greaterThanOrEqualTo': greaterThanOrEqualTo,
        'equalTo': equalTo,
        'anomalyEvent': anomalyEvent,
        'lessThanSparse': lessThanSparse,
        'lessThanOrEqualToSparse': lessThanOrEqualToSparse,
        'greaterThanSparse': greaterThanSparse,
        'greaterThanOrEqualToSparse': greaterThanOrEqualToSparse,
        'equalToSparse': equalToSparse,
        'anomalyEventSparse': anomalyEventSparse
    }
    thisClass = inequalities[className]
    return thisClass

def getFortraccHierarchy(filename):
    """
    Function to read a ForTraCC Mask Output file and create a json 
    file of the internal hierarchy.  Assumes file structure as laid out in
    the ForTraCC repo README.md file
    :param filename: filename of the ForTraCC mask output file, with path
    :type filename: str
    :return fortraccHierarchyFile: filename of the JSON hierarchy file, with path
    :type fortraccHierarchyFile: str
    """
    jsonFilename = filename.split('.')[0] + '-Hierarchy.json'
    ncFile = nc.Dataset(filename, 'r')
    hierarchy = {}

    hierarchy['navigation']  = ['lat','lon']
    hierarchy['masks'] = {}

    for t in ncFile['masks'].groups:
        hierarchy['masks'][t] = ['mask_indices']
    
    with open(jsonFilename, 'w') as outfile:
        json.dump(hierarchy, outfile)

    return jsonFilename

def getCurationHierarchy(jobID, chunkID, info):
    """
    Function to read a curated file and create a json 
    file of the internal hierarchy.  This will be laid out in the same way
    as in the ForTraCC repo README.md file
    :param jobID: jobID of the curation job
    :type jobID: int
    :param chunkID: chunkID of the curation job
    :type jobID: int
    :param info: the dictionary of group names and variables from the curator
    :type info: dict
    :return jsonFilename: filename of the JSON hierarchy file, with path
    :type jsonFilename: str
    """
    if chunkID == None:
        jsonFilename = '/data/tmp/%s-Curation-Hierarchy.json' % (jobID)
    else:
        jsonFilename = '/data/tmp/%s-%s-Curation-Hierarchy.json' % (jobID, chunkID)

    info['navigation'] = ['lat', 'lon']

    with open(jsonFilename, 'w') as outfile:
        json.dump(info, outfile)

    return jsonFilename

def getInterpolationHierarchy(jobID, chunkID, info):
    """
    Function to read a interpolated file and create a json 
    file of the internal hierarchy.  This will be laid out in the same way
    as in the ForTraCC repo README.md file
    :param jobID: jobID of the curation job
    :type jobID: int
    :param chunkID: chunkID for the job
    :type chunkID: int
    :param info: the dictionary of group names and variables from the curator
    :type info: dict
    :return jsonFilename: filename of the JSON hierarchy file, with path
    :type jsonFilename: str
    """
    if chunkID == None:
        jsonFilename = '/data/tmp/%s-Interpolation-Hierarchy.json' % (jobID)
    else:
        jsonFilename = '/data/tmp/%s-%s-Interpolation-Hierarchy.json' % (jobID, chunkID)

    info['navigation'] = ['lat', 'lon']

    with open(jsonFilename, 'w') as outfile:
        json.dump(info, outfile)

    return jsonFilename


def get_json(filename):
    """
    Gets data from JSON files and returns it in dict format
    :param filename: filename of the JSON dictionary
    :type filename: str
    :return j: JSON data from the file
    :rtype j: dict
    """
    fs = s3fs.S3FileSystem()
    with fs.open(filename, 'r') as f:
        j = json.load(f)
    return j

def padTimestamps(timestamps, intervalInfo, first=False, last=False):
    """
    Takes the timestamp info from the PhDef heirarcy file and 
    add a timestamp at the beginning and end, to return an
    udated dict.
    :param timestamps: timestamps from hierarch file
    :type timestamps: dict
    :param intervalInfo: indicates months, days, hours, or minutes and quantity of each like {'units':'minutes', 'quantity': 10}
    :type intervalInfo: dict
    :param fist: include a new first timestamp
    :type first: bool
    :param
    :rtype timestampsNew: timestamps with an additional timestamp at the beginning and end
    :type timestampsNew: dict 
    """
    firstTimestamp = datetime.strptime(list(timestamps.keys())[0], '%Y%m%d%H%M')
    lastTimestamp = datetime.strptime(list(timestamps.keys())[-1], '%Y%m%d%H%M')
    if intervalInfo['units'] == 'months':
        timeChange = relativedelta(months=intervalInfo['quantity'])
    elif intervalInfo['units'] == 'days':
        timeChange = timedelta(days=intervalInfo['quantity'])
    elif intervalInfo['units'] == 'hours':
        timeChange = timedelta(hours=intervalInfo['quantity'])
    elif intervalInfo['units'] == 'minutes':
        timeChange = timedelta(minutes=intervalInfo['quantity'])
    else:
        exit('Bad time interval name')
    newFirst = (firstTimestamp - timeChange).strftime('%Y%m%d%H%M')
    newLast = (lastTimestamp + timeChange).strftime('%Y%m%d%H%M')
    newFirstDict = { newFirst: ['mask_indices']}
    newLastDict = { newLast: ['mask_indices']}
    timestampsNew = {}
    if first:
        timestampsNew.update(newFirstDict)
    timestampsNew.update(timestamps)
    if last:
        timestampsNew.update(newLastDict)

    return timestampsNew

def gridPolygons(lat, lon, latResolution, lonResolution):
    """
    Function (somewhat deprecated) to construct polygons for grid cells
    :param lat: centroid latitude
    :type lat: int
    :param lon: centroid longitude
    :type lon: int
    :param latResolution: the north/south resolution of the grid
    :type latResolution: float
    :param lonResolution: the east/west resolution of the grid
    :type lonResolution: float
    :return polygon: polygon of the grid cell
    :rtype polygon: shapely.geometry.polygon.Polygon
    """

    latNorth = lat + latResolution
    latSouth = lat - latResolution
    lonEast = lon - lonResolution
    lonWest = lon + lonResolution

    topRight = '%s %s' % (lonWest, latNorth)
    bottomRight = '%s %s' % (lonWest, latSouth)
    bottomLeft = '%s %s' % (lonEast, latSouth)
    topLeft = '%s %s' % (lonEast, latNorth)

    polygonString = 'POLYGON((%s, %s, %s, %s, %s))' % (topRight, bottomRight, bottomLeft, topLeft, topRight)
    polygon = wkt.loads(polygonString)

    return polygon

def pushBox(value, mp):
    """
    This function is used by the data curators to ensure that there are enough points
    along the edges of the anomaly bounding box for spatial interpolation.  It pushes the
    bounding box out X degress on each side of the bounding box.  That number of degrees
    is dependant on the spatial resolution of the input dataset.  It returns 'pushed' 
    bounds that are then used to retrieve the data for curation.
    :param value: value, in degrees, that you want to push the bounding box on all 4 sides
    :type value: float
    :param mp: MultiPoint WKT object of the footprint of the anomaly
    :type mp: MultiPoint
    :return minLat: minimum latitude value
    :rtype minLat: float
    :return minLon: minimum longitude value
    :rtype minLon: float
    :return maxLat: maximum latitude value
    :rtype maxLat: float
    :return maxLon: maximum longitude value
    :rtype maxLon: float
    """
    min_lon, min_lat, max_lon, max_lat = mp.bounds
    min_lon = min_lon - value
    min_lat = min_lat - value
    max_lon = max_lon + value
    max_lat = max_lat + value

    return (min_lon, min_lat, max_lon, max_lat)

def timerange(startDate, endDate, interval):
    """ 
    Function to give a list of dates between two dates (inclusive)
    :param startDate: start of the range
    :type startDate: datetime
    :param endDate: end of the range
    :type endDate: datetime
    :param interval: the internval of the data set (hourly, monthly, etc.)
    :type interval: str
    :return dateRange: range of dates
    :rtype dateRange: list of datetimes
    """
    drange = pd.date_range(start=startDate.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), end=endDate.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), freq=interval)
    dateRange = []
    for i in drange:
        dateRange.append(i.to_pydatetime().replace(tzinfo=None))

    return dateRange


def convertLons(minLon, maxLon):
    """
    This function will convert longitude in -180 to 180
    format into 0 to 360, returning the min and lax lons 
    in that format.
    :param minLon: minimum longitude in -180 to 180 format
    :type minLon: float
    :param maxLon: maximum longitude in -180 to 180 format
    :type maxLon: float
    :return convertedMinLon: minimum longitude in 0 to 360 format
    :rtype convertedMinLon: float
    :return convertedMaxLon: maximum longitude in 0 to 360 format
    :type maxLconvertedMaxLon: float
    """
    lonRange = np.arange(minLon, maxLon+1, 1.0)
    convertedLons = []
    for thisLon in lonRange:
        if thisLon == 180.0:
            fixed = 360.0
        else:
            fixed = (thisLon + 360) % 360
        convertedLons.append(fixed)
    
    convertedMinLon = min(convertedLons)
    convertedMaxLon = max(convertedLons)        

    return (convertedMinLon, convertedMaxLon)