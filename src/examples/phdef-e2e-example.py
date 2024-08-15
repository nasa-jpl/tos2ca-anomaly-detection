import sys
from iolib.merra2 import merra2_reader
from utils.plot import mask_plot
from utils.fortracc import callFortracc


if __name__ == '__main__':
    """
    Run this script like:
        % python ./phdef-e2e-example.py <jobID>

    In this example below, we pass in the jobID of a 'phdef' job, that's stored in the database:
        jobID: 390
        userID: 1
        nChunks: 1
        stage: phdef
        phdefJobID: NULL
        dataset: M2I1NXINT_5.12.4
        variable: TQI
        ST_ASTEXT(coords): POLYGON((-103.32 -1.05,-103.32 31.65,-39.34 31.65,-39.34 -1.05,-103.32 -1.05)) 
        startDate: 2020-01-04 00:00:00
        endDate: 2020-02-03 23:59:59
        ineqOperator: anomalyEvent
        ineqValue: 1
        description: Example MERRA-2 PhDef Job
        status: pending
        submitTime: 2024-07-23 19:21:23

    We pass in the jobID (in this case 390), and run that through:
        - the MERRA-2 reader
        - ForTraCC
        
    From that, the PhDef mask netCDF4 file is generated, along with plots and JSON footprints of the anomalies.
    Make sure you specify the AWS S3 bucket name of where you want to store these files.  It will output to:
        s3://<bucket name>/<jobID>/
    """
    jobID = sys.argv[1]
    print(jobID)
    bucketName = 'tos2ca-dev1'
    print("Running jobID: %s" % jobID)
    merra2_reader(jobID)
    print("Job complete")
    print("Run ForTraCC")
    callFortracc(jobID, bucketName)
