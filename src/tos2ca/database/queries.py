from tos2ca.utils import tos2ca_secrets

def getJobInfo(cur, jobID, chunkID=False):
    """
    Function to return all parameters of a submitted job
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    :param chunkID: chunk ID of the submitted job
    :type chunkID: int
    :return results: data from the  select statement
    :type results: dict
    """
    if chunkID != False:
        sql = 'SELECT j.dataset, j.variable, ST_ASTEXT(j.coords) AS coords, c.startDate, c.endDate, j.algorithm, j.ineqOperator, j.ineqValue, j.warmerToggle, j.warmerValue, j.phdefJobID, j.stage, c.status, j.nChunks, c.chunkID FROM jobs j, chunks c WHERE j.jobID=c.jobID AND c.chunkID=%s AND c.jobID=%s' 
        args = (chunkID, jobID)  
    else:
        sql = 'SELECT dataset, variable, ST_ASTEXT(coords) AS coords, startDate, endDate, algorithm, ineqOperator, ineqValue, warmerToggle, warmerValue, phdefJobID, stage, status, nChunks FROM jobs WHERE jobID=%s' 
        args = (jobID)         
    cur.execute(sql, args)
    results = cur.fetchall()

    return results

def getJobChunks(cur, jobID):
    """
    Function to return all chunks for a given jobID in the chunk sql table.
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    :return results: data from the  select statement
    :type results: dict
    """
    sql = 'SELECT chunkID FROM chunks WHERE jobID=%s'
    cur.execute(sql, (jobID))
    results = cur.fetchall()

    return results

def deleteChunks(db, cur, jobID):
    """
    Function to removed chunked curated and interpolated files once the
    stitched file has been created.
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")
    sql = 'SELECT chunkID FROM chunks WHERE jobID=%s'
    cur.execute(sql, (jobID))
    results = cur.fetchall()
    print(results)
    for r in results:
        sql = 'DELETE FROM output WHERE jobID=%s AND location LIKE "s3://%s/%s/%s-%s-%%"'
        cur.execute(sql, (jobID, bucketName, jobID, jobID, r['chunkID']))
    db.commit()

    return

def getFirstChunk(cur):
    """
    Function to return the oldest chunk (<jobID>-<chunkID>) needing to be worked onin the chunk sql table.
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :return results: data from the  select statement
    :type results: dict
    """
    sql = 'SELECT jobID, chunkID, CONCAT(jobID, "-", chunkID) as combinedChunkID FROM chunks where status="pending" ORDER BY jobID asc, chunkID asc LIMIT 1'
    cur.execute(sql)
    results = cur.fetchall()

    return results

def updateStatus(db, cur, jobID, status, chunkID=False, jobStart=False, jobEnd=False):
    """
    Function to update the job status.  
    :param db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    :param status: Valid statuses are: Pending, Submitted, Running, Complete
    :type results: string
    :param chunkID: chunk ID of submitted job; if False, assumed to be job level information
    :type chunkID: int
    :param jobStart: True = updating chunk start time; False = not updating chunk start time
    :type jobStart: bool
    :param jobEnd: True = updating chunk end time; False = not updating chunk end time
    :type jobEnd: bool
    """
    if chunkID == False:
        if status not in ['pending', 'running', 'reading', 
                          'fortracc', 'auxgeoir', 'plotting', 
                          'complete', 'failed']:
            exit('Invalid status')
        
        args = (status, jobID)
        sql = 'UPDATE jobs SET status=%s WHERE jobID=%s'
        
        cur.execute(sql, args)
        db.commit()
    else:
        if status not in ['pending', 'reading', 'fortracc',
                          'subsetting', 'stitching curated',
                          'interpolating', 'stitching interpolation',
                          'complete', 'failed']: 
            exit('Invalid status')
        if jobStart != False:
            args = (status, jobID, chunkID)
            sql = 'UPDATE chunks SET status=%s, jobStart=NOW() WHERE jobID=%s AND chunkID=%s'
        elif jobEnd != False:
            args = (status, jobID, chunkID)
            sql = 'UPDATE chunks SET status=%s, jobEnd=NOW() WHERE jobID=%s AND chunkID=%s'
        else:
            args = (status, jobID, chunkID)
            sql = 'UPDATE chunks SET status=%s WHERE jobID=%s AND chunkID=%s'

        cur.execute(sql, args)
        db.commit()
    return

def insertOutputFile(db, cur, jobID, filename):
    """
    Function to update the job status.  
    :param db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    :param filename: filenames for a job
    :type filename: list
    """
    sql = 'INSERT INTO output SET jobID=%s, location=%s'
    cur.execute(sql, (jobID, filename))
    db.commit()

    return
