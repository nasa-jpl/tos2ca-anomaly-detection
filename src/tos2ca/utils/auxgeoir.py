import json
import time

from tos2ca.database.connection import openDB, closeDB
from tos2ca.database.elasticache import getData, setAuxGeoIRData, getAuxGeoIRData
from tos2ca.database.queries import updateStatus
from fortracc_module.objects import SparseGeoGrid
from fortracc_module.utils import write_nc4
from fortracc_module.chunking import stitch
from auxgeoir_module.processing import run_storm_tracking_pipeline_for_tos2ca
from tos2ca.utils.helpers import getAuxGeoIRHierarchy
from tos2ca.utils.s3 import s3Upload
from tos2ca.utils import tos2ca_secrets


def callAuxGeoIRSparse(jobID, chunkID):
    """
    Call and run AUX-GEOIR using the sparse methods, usually with chunks
    Note that the warmer threshold is set below, and is not taken as a user
    input at this time.  Additionally, you should only use the equals ('=')
    operator/detector at this time.  This borrows functions from the ForTraCC 
    library.
    :param jobIDs: job ID
    :type jobID: int
    :param chunkID: chunk ID of the job
    :type chunkID: int
    :param bucketName: name of the AWS S3 bucket to write output to
    :type bucketName: str
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'auxgeoir')
    closeDB(db)

    auxgeoir_inputs = []
    data, jobInfo, startDateTime = getData(jobID, chunkID)
    g = SparseGeoGrid.from_lat_lon(data['lat'], data['lon'])
    if not jobInfo['ineqOperator'].endswith("Sparse"):
        jobInfo['ineqOperator'] += "Sparse"
    timestamps, images = list(zip(*data['images'].items()))

    # Detector is always be less than ('<')
    # default thresholds are set to match ForTraCC as best as possible
    auxgeoir_inputs.append({
        "images": images,
        "timestamps": timestamps,
        "grid": g,
        "temp_thresh": jobInfo['ineqValue'],
        "temp_warmer_thresh": jobInfo['warmerValue'],
        "min_size": 150,
        "max_size_threshold": 2500,
        "overlap_percentage": 0.25,
        "toggle": jobInfo['warmerToggle']
    })

    results = []
    for inputs in auxgeoir_inputs:
        s = time.time()
        results.append(run_storm_tracking_pipeline_for_tos2ca(**inputs))
        e = time.time()
        print(f'Elapsed time: {e - s:.4f}s')

    if len(results) == 1:
        setAuxGeoIRData(results[0], jobID, chunkID)
    return


def stitchAuxGeoIR(jobID):
    """
    Function to stitch together results of AUX-GEOIR runs
    that are being stored in Elasticache into a single
    netCDF-4 file.  This borrows functions from the ForTraCC library.
    :param jobID: job ID to run
    :type jobID: int
    """
    db, cur = openDB()
    updateStatus(db, cur, jobID, 'auxgeoir')
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
        auxgeoirData = getAuxGeoIRData(jobID, thisChunk['chunkID'])
        results.append(auxgeoirData)
    stos = stitch(results)
    print('Writing netCDF output...')
    metadata = {'jobID': jobID, 'variable': jobInfo['variable'], 'dataset': jobInfo['dataset'], 'threshold': str(jobInfo['ineqValue'])}
    anomaly_table = write_nc4(stos, f'{jobID}-AuxGeoIR-Mask-Output.nc4', output_dir='/data/tmp', metadata=metadata)
    print('Writing JSON table of contents...')
    toc = json.dumps(anomaly_table)
    with open('/data/tmp/' + str(jobID) + '-AuxGeoIR-TOC.json', 'w') as f:
        f.write(toc)
    print('Uploading TOC file to S3...')
    db, cur = openDB()
    jobInfo = {'filename': f'/data/tmp/{jobID}-AuxGeoIR-TOC.json',
                'startDateTime': startDateTime,
                'type': 'toc'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Creating and uploading hierarchy JSON file...')
    jsonFilename = getAuxGeoIRHierarchy(f'/data/tmp/{jobID}-AuxGeoIR-Mask-Output.nc4')
    jobInfo = {'filename': jsonFilename,
                'startDateTime': startDateTime,
                'type': 'hierarchy'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    print('Uploading nc4 Mask file to S3...')
    jobInfo = {'filename': f'/data/tmp/{jobID}-AuxGeoIR-Mask-Output.nc4',
                'startDateTime': startDateTime,
                'type': 'masks'}
    s3Upload(jobID, jobInfo, bucketName, db, cur)
    closeDB(db)

    return
