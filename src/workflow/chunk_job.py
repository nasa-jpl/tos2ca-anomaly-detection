import sys
from database.connection import openDB, closeDB
from database.queries import getJobInfo
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import json
import pymysql

def chunk_job(jobID):
    """
    Function to chunk a large job (in time) into smaller chunks, ideally
    to be run in containers.
    :param jobID: the jobID 
    :type jobID: int
    :
    """

    # Open the database connection and get the info for this runID
    db, cur = openDB()
    info = getJobInfo(cur, jobID)[0]
    print(info)
    # Get start/end dates of the job
    startDate = info['startDate']
    endDate   = info['endDate']
    # Set up date looping
    datesInJob = pd.date_range(start=startDate, end=endDate)
    nDays = len(datesInJob)
    print('xx  ', datesInJob[0].strftime("%Y-%m-%d"))
    # Load in the data dictionary that tells what we are reading in
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as j:
        dataDict = json.load(j)
    timeStep = dataDict[info['dataset']]['timeStep']

    sqlStmts = []
    if timeStep == 'monthly':

        # Figure out the distinct months we want
        distinct_months = datesInJob.to_series().dt.to_period('M').unique()
        distinct_months = sorted(pd.to_datetime(distinct_months.to_timestamp()))
        nChunks = max(len(distinct_months)-2, 1)
        thisMonth = distinct_months[0]
        i = 0
        while i < nChunks:
            if distinct_months[0] == distinct_months[-1]:
                tmp = (jobID, 1, startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d'))
            else:
                tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d'), distinct_months[i+1].strftime('%Y-%m-%d'))
            print(tmp)
            sqlStmts.append('INSERT INTO Chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
            i = i + 1
    elif timeStep == 'daily':
        # To make the stitching work, the last day does not need to be a distinct chunk
        nChunks = max(nDays-2, 1)
        if daysInJob == 1:
            sqlStmts.append('INSERT INTO Chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % (
                jobID, 1, startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')
            ))
        else:
            i = 0
            while i < nChunks:
                sqlStmts.append('INSERT INTO Chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % (
                    jobID, i+1, datesInJob[i].strftime('%Y-%m-%d'), datesInJobs[i+1].strftime('%Y-%m-%d')
                ))
                i = i + 1
    elif info['dataset'] == 'OISSS_L4_multimission_7day_v1':
        # Don't know how to do these
        print("Can't chunk OISSS yet")
    else:
        # Less than hourly chunk by day (last chunk will be 2 days)
        if nDays == 1:
            tmp = (jobID, i+1, datesInJob[0].strftime('%Y-%m-%d'), datesInJob[0].strftime('%Y-%m-%d'))
        else:
            nChunks = nDays
            i = 0
            print(startDate, endDate, nChunks)
            
            while i < nChunks:
                if i == (nDays-1):
                    tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d'), datesInJob[i].strftime('%Y-%m-%d'))
                else:
                    tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d'), datesInJob[i+1].strftime('%Y-%m-%d'))
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
                i = i + 1
    #    
    #sqlStmts.append('UPDATE jobs SET status="running", nChunks=%s WHERE jobID=%s' % (nChunks, jobID)) # :TODO: Replace when in production
    sqlStmts.append('UPDATE jobs SET nChunks=%s WHERE jobID=%s' % (nChunks, jobID))

    try:
        for sql in sqlStmts:
            print(sql)
            cur.execute(sql)
        db.commit()
    except pymysql.MySQLError as e:
        print(f"Error: {e}")
        db.rollback()
    finally:    
        closeDB(db)
    
    return


if __name__ == '__main__':
 
    if len(sys.argv) != 2:
        print("Usage python chunk_job.py <jobID>")
    jobID = sys.argv[1]

    chunk_job(jobID)  
