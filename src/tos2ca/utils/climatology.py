import xarray as xr
import json
import netCDF4 as nc
import numpy as np
import math
import s3fs

from utils import tos2ca_secrets
from utils.s3 import s3Upload
from datetime import datetime, timedelta
from database.connection import openDB, closeDB
from glob import glob
from utils.helpers import get_json


def fullJSON(jobID):
    """
    This function will look for any curation jobs associated
    with a PhDef run.  It will then grab all the datetimes
    and variable names assoicated with those runs and create
    a JSON file strucutre.  Then individual files will be read in
    to populate that file.
    This JSON file can easily be imported into a Pandas data frame.
    It is mostly commonly used on jobs generated from the pre-defined
    PhDef stage, where there are longer periods of data that were generated
    using the 'stationary' curators.
    This should only be used on full interpolated files (not chunked files).
    :param jobID:  PhDef jobID
    :type jobID: int
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    jsonFilename = '/data/tmp/%s-Climatology.json' % (jobID)

    db, cur = openDB()

    sql = 'SELECT DISTINCT jobID, variable FROM jobs WHERE phdefJobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchall()

    variables = []
    curationJobIDs = []
    for r in results:
        variables.append(r['variable'])
        curationJobIDs.append(r['jobID'])

    dates = []

    sql = 'SELECT startDate, endDate FROM jobs WHERE jobID=%s'
    cur.execute(sql, jobID)
    results = cur.fetchone()
    startDate = results['startDate'] - timedelta(days=1)
    endDate = results['endDate'] + timedelta(days=1)
    currentDate = startDate
    while currentDate <= endDate:
        dates.append(currentDate.date())
        currentDate += timedelta(days=1)

    closeDB(db)

    info = {}
    info['jobID'] = jobID
    info['startDate'] = startDate.strftime('%Y-%m-%d %H:%M:%S')
    info['endDate'] = endDate.strftime('%Y-%m-%d %H:%M:%S')
    info['variables'] = variables
    info['data'] = {}

    temp = {}
    for d in dates:
        temp[d.strftime('%Y%m%d')] = {}

    info['data']['columns'] = []
    info['data']['rows'] = []

    for d in dates:   
        for v in variables:
            temp[d.strftime('%Y%m%d')][f'{v}_min'] = []
            temp[d.strftime('%Y%m%d')][f'{v}_max'] = []
            temp[d.strftime('%Y%m%d')][f'{v}_mean'] = []

    print(temp)

    for ch in curationJobIDs:
        db, cur = openDB()

        sql = 'SELECT location FROM output WHERE jobID=%s AND location="s3://%s/%s/%s-Interpolated-Data.nc4" AND type="interpolated subset"' % (ch, bucketName, ch, ch)
        cur.execute(sql)
        result = cur.fetchone()
        interpolatedFile = result['location']

        sql = 'SELECT location FROM output WHERE jobID=%s AND location="s3://%s/%s/%s-Interpolation-Hierarchy.json" AND type="interpolated hierarchy"' % (ch, bucketName, ch, ch)
        cur.execute(sql)
        result = cur.fetchone()
        hierarchyFile = result['location']

        closeDB(db)
        
        fs = s3fs.S3FileSystem()
   
        j = get_json(hierarchyFile)
        del j['navigation']

        dateKeys = sorted(j.keys())
        
        product = j[dateKeys[0]]['1'][0]

        for d in dateKeys:
            print(f'{d}/1/{product}')
            ds = xr.open_dataset(fs.open(interpolatedFile, 'rb'), group = d + '/1')
            dkString = d[0:8]
            temp[dkString][f'{product}_mean'].append(float(ds[product].Mean.strip()))
            temp[dkString][f'{product}_min'].append(float(ds[product].Min.strip()))
            temp[dkString][f'{product}_max'].append(float(ds[product].Max.strip()))

    firstKey = next(iter(temp))
    valueKeys = list(temp[firstKey].keys())
    info['data']['columns'].append('datetime')
    for v in valueKeys:
        info['data']['columns'].append(v)

    for d in temp:
        placeholder = []
        placeholder.append(datetime.strptime(d, '%Y%m%d').timestamp())
        for k in temp[d]:
            if '_mean' in k:
                thisValue = np.mean(temp[d][k])
            elif '_max' in k:
                if len(temp[d][k]) == 0:
                    thisValue = np.nan
                else:
                    thisValue = np.max(temp[d][k])
            elif '_min' in k:
                if len(temp[d][k]) == 0:
                    thisValue = np.nan
                else:
                    thisValue = np.min(temp[d][k])
            
            if math.isnan(thisValue):
                placeholder.append(None)
            else:
                placeholder.append(thisValue)
        info['data']['rows'].append(placeholder)

    print('Writing %s' % jsonFilename)

    print(info)

    with open(jsonFilename, 'w') as f:
        json.dump(info, f)

    jobInfo = {
        'filename': jsonFilename,
        'startDateTime': startDate,
        'type': 'JSON climatology'
    }
  
    db, cur = openDB()
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    closeDB(db)

    return