"""
These functions are meant to be examples of ways that you can 
chunk jobs for different time periods for both the PhDef and 
Data Curation stages.  You can chunk the data to any sort of 
temporal granulairty you want.  Remember that for PhDef, 
FortraCC will need an overlapping chunks.  Also remember that the
max object size for ValKey storage in the PhDef stage is 512MB. 
Make sure you chunk with data size in mind for PhDef.
"""

import pandas as pd
import json
import pymysql
from tos2ca.database.connection import openDB, closeDB
from tos2ca.database.queries import getJobInfo
from pandas.tseries.offsets import MonthEnd
from datetime import timedelta

def chunk_job(jobID):
    """
    Function to chunk a large job (in time) into smaller chunks, ideally
    to be run in containers.
    :param jobID: the jobID 
    :type jobID: int
    """

    # Open the database connection and get the info for this runID
    db, cur = openDB()
    info = getJobInfo(cur, jobID)[0]
    print(info)
    # Get start/end dates of the job
    if info['stage'] == 'phdef':
        startDate = info['startDate']
        endDate = info['endDate']
        dataset = info['dataset']
    elif info['stage'] == 'curation':
        phdefJobInfo = getJobInfo(cur, info['phdefJobID'])[0]
        startDate = phdefJobInfo['startDate']
        endDate = phdefJobInfo['endDate']
    else:
        exit('Invalid stage for chunking.')
    # Set up date looping
    datesInJob = pd.date_range(start=startDate, end=endDate)
    nDays = len(datesInJob)
    print('xx  ', datesInJob[0].strftime("%Y-%m-%d"))
    # Load in the data dictionary that tells what we are reading in
    with open('/data/code/data-dictionaries/tos2ca-phdef-dictionary.json') as j:
        dataDict = json.load(j)
    if info['stage'] == 'phdef':
        timeStep = dataDict[info['dataset']]['timeStep']
    elif info['stage'] == 'curation':
        phdefJobInfo = getJobInfo(cur, info['phdefJobID'])[0]
        timeStep = dataDict[phdefJobInfo['dataset']]['timeStep']
    else:
        exit('Invalid stage for chunking.')

    sqlStmts = []
    # Monthly Data
    if timeStep == 'monthly':
        # Figure out the distinct months we want
        distinct_months = datesInJob.to_series().dt.to_period('M').unique()
        distinct_months = sorted(pd.to_datetime(distinct_months.to_timestamp()))
        nChunks = max(len(distinct_months)-2, 1)
        if distinct_months[0] == distinct_months[-1]:
            if info['stage'] == 'phdef':
                tmp = (jobID, 1, startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d'))
            if info['stage'] == 'curation':
                tmp = (jobID, 1, startDate.strftime('%Y-%m-%d 00:00:00'), endDate.strftime('%Y-%m-%d 23:59:59'))
            sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
        else:
            nChunks = len(distinct_months)
            nDays = nChunks
            i = 0
            while i < nChunks:
                # deal with the last day
                if i == (nDays-1):
                    if info['stage'] == 'phdef':
                        endOfMonth = distinct_months[i] + MonthEnd(1)
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d %H:%M:%S'), endOfMonth.strftime('%Y-%m-%d 23:59:59'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d 00:00:00'), distinct_months[i].strftime('%Y-%m-%d 23:59:59'))                        
                # deal with the first day
                elif i == 0:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d %H:%M:%S'), distinct_months[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d 00:00:00'), distinct_months[i].strftime('%Y-%m-%d 23:59:59'))                    
                # deal with all other days
                else:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d %H:%M:%S'), distinct_months[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, distinct_months[i].strftime('%Y-%m-%d 00:00:00'), distinct_months[i].strftime('%Y-%m-%d 23:59:59'))
                print(tmp)
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
                i = i + 1
    # Daily Data
    elif timeStep == 'daily':
        # To make the stitching work, the last day does not need to be a distinct chunk
        nChunks = max(nDays-2, 1)
        if nChunks == 1:
            if info['stage'] == 'phdef':
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % (
                    jobID, 1, startDate.strftime('%Y-%m-%d'), endDate.strftime('%Y-%m-%d')
                ))
            if info['stage'] == 'curation':
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % (
                    jobID, 1, startDate.strftime('%Y-%m-%d 00:00:00'), endDate.strftime('%Y-%m-%d 23:59:59')
                ))
        else:
            nChunks = nDays
            i = 0
            print(datesInJob)
            while i < nChunks:
                # deal with the last day
                if i == (nDays-1):
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                        
                # deal with the first day
                elif i == 0:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                    
                # deal with all other days
                else:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
                i = i + 1
    # Note that OISS Data  where a week is 4 days (and only has PhDef)
    # Note that SEA_SURFACE has a 5 day week (and only has PhDef)
    # Because of the irregularity of the dates and the need for the overlap, 
    # we're only going to allow these data sets' jobs to be 1 chunk since the files are so small
    elif timeStep == 'weekly':
        nChunks = 1
        sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % (
                jobID, 1, startDate.strftime('%Y-%m-%d 00:00:00'), endDate.strftime('%Y-%m-%d 23:59:59')))
    # Since it's half hourly, we have to get pad the start time with the last half hour of
    # the previous day in PhDef or ForTraCC will get confused.
    elif timeStep == 'half hourly':
        # If there's only one day we're working with
        if nDays == 1:
            nChunks = 1
            if info['stage'] == 'phdef':
                tmp = (jobID, 1, datesInJob[0].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[0].strftime('%Y-%m-%d'))
            if info['stage'] == 'curation':
                tmp = (jobID, 1, datesInJob[0].strftime('%Y-%m-%d 00:00:00'), datesInJob[0].strftime('%Y-%m-%d 23:59:59'))                
            sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
        # If there's more than one day
        else:
            nChunks = nDays
            i = 0
            print(datesInJob)
            while i < nChunks:
                # deal with the last day
                if i == (nDays-1):
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                        
                # deal with the first day
                elif i == 0:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                    
                # deal with all other days
                else:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
                i = i + 1
    elif timeStep == 'hourly':
        # If there's only one day we're working with
        if nDays == 1:
            nChunks = 1
            if info['stage'] == 'phdef':
                tmp = (jobID, 1, datesInJob[0].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[0].strftime('%Y-%m-%d'))
            if info['stage'] == 'curation':
                tmp = (jobID, 1, datesInJob[0].strftime('%Y-%m-%d 00:00:00'), datesInJob[0].strftime('%Y-%m-%d 23:59:59'))                
            sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
        # If there's more than one day
        else:
            nChunks = nDays
            i = 0
            print(datesInJob)
            while i < nChunks:
                # deal with the last day
                if i == (nDays-1):
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                        
                # deal with the first day
                elif i == 0:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))                    
                # deal with all other days
                else:
                    if info['stage'] == 'phdef':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d %H:%M:%S'), datesInJob[i+1].strftime('%Y-%m-%d'))
                    if info['stage'] == 'curation':
                        tmp = (jobID, i+1, datesInJob[i].strftime('%Y-%m-%d 00:00:00'), datesInJob[i].strftime('%Y-%m-%d 23:59:59'))
                sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
                i = i + 1
    else:
        # We are skipping ASCAT for now but might revisit later
        exit('Cannot work with time step %s' % timeStep)
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


def hourly_chunks(jobID):
    '''
    This will chunk a curation job into hourly chunks. 
    May later chunk PhDef into hourly chunks, but leaves it alone for now.
    :param jobID: the jobID to chunk
    :type jobID: int
    '''
    # Open the database connection and get the info for this runID
    db, cur = openDB()
    info = getJobInfo(cur, jobID)[0]
    print(info)
    # Get start/end dates of the job
    if info['stage'] == 'phdef':
        startDate = info['startDate']
        endDate = info['endDate']
    elif info['stage'] == 'curation':
        phdefJobInfo = getJobInfo(cur, info['phdefJobID'])[0]
        startDate = phdefJobInfo['startDate']
        endDate = phdefJobInfo['endDate']
    else:
        exit('Invalid stage for chunking.')
    # Set up date looping
    hoursInJob = pd.date_range(start=startDate, end=endDate, freq='H')
    nHours = len(hoursInJob)
    print('xx  ', hoursInJob[0].strftime("%Y-%m-%d"))

    sqlStmts = []

    nChunks = nHours
    i = 0
    while i < nChunks:
        #if 'GPM_' in info['dataset']:
        #    tmp = (jobID, i+1, hoursInJob[i].strftime('%Y-%m-%d %H:%M:%S'), (hoursInJob[i] + timedelta(hours=1)).strftime('%Y-%m-%d %H:00:00'))
        #else:
        tmp = (jobID, i+1, hoursInJob[i].strftime('%Y-%m-%d %H:%M:%S'), hoursInJob[i].strftime('%Y-%m-%d %H:59:59'))
        sqlStmts.append('INSERT INTO chunks (jobID, chunkID, status, startDate, endDate) VALUES (%s, %s, "pending", "%s", "%s")' % tmp)
        i = i + 1

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