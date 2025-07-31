import sys
from iolib.merra2 import merra2_reader
from utils.plot import mask_plot
from utils.fortracc import callFortraccSparse, stitchFortracc


if __name__ == '__main__':
    """
    Run this script like:
        % python ./phdef-e2e-example.py <jobID> <chunkID>

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
        - plotting program

    From that, the PhDef mask netCDF-4 file is generated, along with plots and JSON footprints of the anomalies.
    It will output to the S3 bucket name that you have identified in AWS S3:
        s3://<bucket name>/<jobID>/
    """
    jobID = sys.argv[1]
    chunkID = sys.argv[2]
    print("Running jobID: %s-%s" % (jobID, chunkID))
    merra2_reader(jobID, chunkID)
    print("Job complete")
    print("Run ForTraCC")
    callFortraccSparse(jobID, chunkID)
    print("Combine chunks into a single file")
    stitchFortracc(jobID)
    print("Done with ForTraCC")
    print("Plot the data")
    mask_plot(jobID, chunkID)
    print("Done plotting")
