import json
import netCDF4 as nc
import s3fs
import shapely.wkt as wkt

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

def getCurationHierarchy(jobID, info):
    """
    Function to read a curated file and create a json 
    file of the internal hierarchy.  This will be laid out in the same way
    as in the ForTraCC repo README.md file
    :param jobID: jobID of the curation job
    :type filename: int
    :param info: the dictionary of group names and variables from the curator
    :type info: dict
    :return fortraccHierarchyFile: filename of the JSON hierarchy file, with path
    :type fortraccHierarchyFile: str
    """
    jsonFilename = '/data/tmp/%s-Curation-Hierarchy.json' % jobID

    info['navigation'] = ['lat', 'lon']

    with open(jsonFilename, 'w') as outfile:
        json.dump(info, outfile)

    return jsonFilename

def get_json(filename):
    """
    Gets data from JSON files and returns it in dict format
    :param filename: filename of the JSON dictionary
    :type jobID: str
    :return j: JSON data from the file
    :rtype j: dict
    """
    fs = s3fs.S3FileSystem()
    with fs.open(filename, 'r') as f:
        j = json.load(f)
    return j

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
    along the edges of the anomly bounding box for spatial interpolation.  It pushes the
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
