import json
import time

from tos2ca.database.connection import openDB, closeDB
from tos2ca.database.elasticache import getData, setFortraccData, getFortraccData
from tos2ca.database.queries import updateStatus
from fortracc_module.objects import GeoGrid, SparseGeoGrid
from fortracc_module.utils import write_nc4
from fortracc_module.chunking import stitch
from fortracc_module.flow import SparseTimeOrderedSequence
from tos2ca.utils.helpers import getOperatorClass, getFortraccHierarchy
from tos2ca.utils.s3 import s3Upload
from tos2ca.utils import tos2ca_secrets


def callFortracc(jobID, bucketName, chunkID):
    """
    Call and run ForTracc for a single job (no chunks)
    :param jobID: job ID to process
    :type jobID: int
    :param chunkID: chunk ID to process
    :type chunkID: int
    :param bucketName: name of the AWS S3 bucket to write output to
    :type bucketName: str
    """
    data, jobInfo, startDateTime = getData(jobID, chunkID)
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'fortracc', chunkID=chunkID)
    closeDB(db)

    fortracc_inputs = {
        "name": jobInfo['dataset'],
        "lat": data['lat'],
        "lon": data['lon'],
        "images": data['images'],
        "inequality": getOperatorClass(jobInfo['ineqOperator']),
        "threshold": jobInfo['ineqValue']
    }
    g = GeoGrid(fortracc_inputs['lat'], fortracc_inputs['lon'])
    timestamps, images = list(zip(*fortracc_inputs['images'].items()))
    tos = fortracc_inputs['inequality'](
        images, timestamps, g, fortracc_inputs['threshold'])
    tos.run_fortracc()
    print('Writing netCDF output...')
    if 'anomalyEvent' in jobInfo['ineqValue']:
        metadata = {'jobID': jobID, 'variable': jobInfo['variable'], 'dataset': jobInfo['dataset'], 'frac_std': str(jobInfo['ineqValue'])}
    else:
        metadata = {'jobID': jobID, 'variable': jobInfo['variable'], 'dataset': jobInfo['dataset'], 'threshold': str(jobInfo['ineqValue'])}
    anomaly_table = write_nc4(tos, f'{jobID}-ForTraCC-Mask-Output.nc4', output_dir='/data/tmp', metadata=metadata)
    print('Writing JSON table of contents...')
    toc = json.dumps(anomaly_table)
    with open('/data/tmp/' + str(jobID) + '-ForTraCC-TOC.json', 'w') as f:
        f.write(toc)
    print('Uploading TOC file to S3...')
    db, cur = openDB()
    jobInfo = {'filename': f'/data/tmp/{jobID}-ForTraCC-TOC.json',
                'startDateTime': startDateTime,
                'type': 'toc'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Creating and uploading hierarchy JSON file...')
    jsonFilename = getFortraccHierarchy(f'/data/tmp/{jobID}-ForTraCC-Mask-Output.nc4')
    jobInfo = {'filename': jsonFilename,
                'startDateTime': startDateTime,
                'type': 'hierarchy'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Uploading nc4 Mask file to S3...')
    jobInfo = {'filename': f'/data/tmp/{jobID}-ForTraCC-Mask-Output.nc4',
                'startDateTime': startDateTime,
                'type': 'masks'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    closeDB(db)

    return


def callFortraccSparse(jobID, chunkID):
    """
    Call and run ForTracc using the sparse methods, usually with chunks
    :param jobIDs: job ID
    :type jobID: int
    :param chunkID: chunk ID of the job
    :type chunkID: int
    :param bucketName: name of the AWS S3 bucket to write output to
    :type bucketName: str
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'fortracc')
    closeDB(db)

    fortracc_inputs = []
    data, jobInfo, startDateTime = getData(jobID, chunkID)
    g = SparseGeoGrid.from_lat_lon(data['lat'], data['lon'])
    if not jobInfo['ineqOperator'].endswith("Sparse"):
        jobInfo['ineqOperator'] += "Sparse"
    detectorType = getOperatorClass(jobInfo['ineqOperator'])
    if 'anomalyEvent' in jobInfo['ineqOperator']:
        detector = detectorType(frac_std=jobInfo['ineqValue'])
    else:
        detector = detectorType(threshold=jobInfo['ineqValue'])
    timestamps, images = list(zip(*data['images'].items()))
    fortracc_inputs.append({
        "images": images,
        "timestamps": timestamps,
        "grid": g,
        "detector": detector,
        "connectivity": 2,
        "min_olap": 0.25,
        "min_size": 150,
    })
    results = []
    for inputs in fortracc_inputs:
        s = time.time()
        results.append(SparseTimeOrderedSequence.run_fortracc(**inputs))
        e = time.time()
        print(f'Elapsed time: {e - s:.4f}s')

    if len(results) == 1:
        setFortraccData(results[0], jobID, chunkID)

    return


def stitchFortracc(jobID):
    """
    Function to stitch together results of FortraCC runs
    that are being stored in Elasticache into a single
    netCDF-4 file
    :param jobID: job ID to run
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'fortracc')
    sql = 'SELECT chunkID FROM chunks WHERE jobID=%s ORDER BY chunkID ASC'
    cur.execute(sql, (jobID))
    chunkList = cur.fetchall()
    closeDB(db)
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")

    data, jobInfo, startDateTime = getData(jobID, 1)
    del data
    results = []
    for thisChunk in chunkList:
        fortraccData = getFortraccData(jobID, thisChunk['chunkID'])
        results.append(fortraccData)
    stos = stitch(results)
    print('Writing netCDF output...')
    metadata = {'jobID': jobID, 'variable': jobInfo['variable'], 'dataset': jobInfo['dataset'], 'threshold': str(jobInfo['ineqValue'])}
    anomaly_table = write_nc4(stos, f'{jobID}-ForTraCC-Mask-Output.nc4', output_dir='/data/tmp', metadata=metadata)
    print('Writing JSON table of contents...')
    toc = json.dumps(anomaly_table)
    with open('/data/tmp/' + str(jobID) + '-ForTraCC-TOC.json', 'w') as f:
        f.write(toc)
    print('Uploading TOC file to S3...')
    db, cur = openDB()
    jobInfo = {'filename': f'/data/tmp/{jobID}-ForTraCC-TOC.json',
                'startDateTime': startDateTime,
                'type': 'toc'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Creating and uploading hierarchy JSON file...')
    jsonFilename = getFortraccHierarchy(f'/data/tmp/{jobID}-ForTraCC-Mask-Output.nc4')
    jobInfo = {'filename': jsonFilename,
                'startDateTime': startDateTime,
                'type': 'hierarchy'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Uploading nc4 Mask file to S3...')
    jobInfo = {'filename': f'/data/tmp/{jobID}-ForTraCC-Mask-Output.nc4',
                'startDateTime': startDateTime,
                'type': 'masks'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    closeDB(db)

    return
