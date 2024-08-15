import sys
from iolib.gpm import gpm_curator
from utils.interpolation import interpolator


if __name__ == '__main__':
    """
    Run this script like:
        % python ./curation-e2e-example.py <jobID>

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

    We pass in the jobID (in this case 390), and run that through:
        - the GPM curator
        - the interpolator
        
    From that, the curated and interpolated mask netCDF4 files are generated.
    Make sure you specify the AWS S3 bucket name of where you want to store these files.  It will output to:
        s3://<bucket name>/<jobID>/
    """
    jobID = sys.argv[1]
    print(jobID)
    print("Running jobID: %s" % jobID)
    gpm_curator(jobID)
    print("Curation complete")
    print("Run Interpolater")
    interpolator(jobID)
    print("Done interpolating")