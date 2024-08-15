def getJobInfo(cur, jobID):
    """
    Function to return all parameters of a submitted job
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    :param jobID: job ID of the submitted job
    :type jobID: int
    :return results: data from the  select statement
    :type results: dict
    """
    if '-' in str(jobID):
        (jobID1, chunkID) = jobID.split('-')
        sql = 'SELECT j.dataset, j.variable, ST_ASTEXT(j.coords) AS coords, c.startDate, c.endDate, j.ineqOperator, j.ineqValue, j.phdefJobID, j.stage, c.status, j.nChunks, c.chunkID FROM jobs j, chunks c WHERE j.jobID=c.jobID AND c.chunkID=%s AND c.jobID=%s' 
        args = (chunkID,jobID1)    
    else:
        sql = 'SELECT dataset, variable, ST_ASTEXT(coords) AS coords, startDate, endDate, ineqOperator, ineqValue, phdefJobID, stage, status, nChunks FROM jobs WHERE jobID=%s'
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

def updateStatus(db, cur, jobID, status):
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
    """
    if status not in ['pending', 'running', 'complete', 'error']:
        exit('Invalid status')
    if '-' in str(jobID):
        (jobID1, chunkID) = jobID.split('-')
        args = (status, jobID1, chunkID)
        sql = 'UPDATE chunks SET status=%s WHERE jobID=%s AND chunkID=%s'
    else:
        args = (status, jobID)
        sql = 'UPDATE jobs SET status=%s WHERE jobID=%s'
    
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
