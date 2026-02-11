import sys
from iolib.gpm import gpm_curator
from utils.interpolation import interpolator
from utils.ncTools import combineCuratedFiles, combineInterpolatedFiles, cleanUpChunks


if __name__ == '__main__':
    """
    Run this script like:
        % python ./curation-e2e-example.py <jobID> <chunkID>

    In this example below, we pass in the jobID of a 'curation' job, that's stored in the database:
        jobID: 399
        userID: 1
        nChunks: 1
        stage: curation
        phdefJobID: 380
        dataset: GPM_MERGIR
        variable: Tb
        coords: NULL
        startDate: NULL
        endDate: NULL
        ineqOperator: NULL
        ineqValue: NULL
        description: NULL
        status: pending
        submitTime: 2024-07-23 20:42:07

    We pass in the jobID (in this case 399), and run that through:
        - the GPM curator
        - the interpolator
        - the stitchers (even though there is only 1 chunk)

    From that, the curated and interpolated mask netCDF4 files are generated.
    Make sure you specify the AWS S3 bucket name of where you want to store these files.  
    It will output to:
        s3://<bucket name>/<jobID>/
    """
    jobID = sys.argv[1]
    chunkID = sys.argv[2]
    print(jobID, chunkID)
    print("Running jobID: %s-%s" % (jobID, chunkID))
    gpm_curator(jobID, chunkID)
    # If you have more than 1 chunk run the combiner commented out below
    # print("Combine chunks into single file")
    # combineCuratedFiles(jobID)
    print("Curation complete")
    print("Run Interpolater")
    interpolator(jobID, chunkID)
    print("Done interpolating")
    # If you have more than 1 chunk run the combiner commented out below
    # print("Combine chunks into single file")
    # combineInterpolatedFiles(jobID)
    print("Clean up chunks")
    cleanUpChunks(jobID)
    print("Finished")
