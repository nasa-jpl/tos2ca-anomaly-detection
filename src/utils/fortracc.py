import json
import time

from database.connection import openDB, closeDB, openCache
from database.elasticache import getData
from database.queries import updateStatus
from fortracc_module.objects import GeoGrid, SparseGeoGrid
from fortracc_module.utils import write_nc4
from fortracc_module.chunking import create_file_chunks, stitch
from fortracc_module.detectors import GreaterThanDetector
from fortracc_module.flow import SparseTimeOrderedSequence
from utils.helpers import getOperatorClass, getFortraccHierarchy
from utils.s3 import s3Upload


def callFortracc(jobID, bucketName):
    """
    Call and run ForTracc
    :param jobID: job ID to process
    :type jobID: int
    :param bucketName: name of the AWS S3 bucket to write output to
    :type bucketName: str
    """
    r = openCache()
    data, jobInfo, startDateTime = getData(r, jobID)
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
    updateStatus(db, cur, jobID, 'complete')
    closeDB(db)
    
    return

def callFortraccSparse(jobIDs, bucketName):
    """
    Call and run ForTracc using the sparse methods
    :param jobIDs: job IDs to process
    :type jobID: list
    :param bucketName: name of the AWS S3 bucket to write output to
    :type bucketName: str
    """
    r = openCache()
    fortracc_inputs = []
    for jobID in jobIDs:
        data, jobInfo, startDateTime = getData(r, jobID)
        g = SparseGeoGrid.from_lat_lon(data['lat'], data['lon'])
        detectorType = getOperatorClass(jobInfo['ineqOperator'])
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
    r = openCache() 
    for inputs in fortracc_inputs:
        s = time.time()
        results.append(SparseTimeOrderedSequence.run_fortracc(**inputs))
        e = time.time()
        print(f'Elapsed time: {e - s:.4f}s')
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
    updateStatus(db, cur, jobID, 'complete')
    closeDB(db)
    
    return
