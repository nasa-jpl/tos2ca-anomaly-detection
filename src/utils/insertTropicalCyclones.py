import pymysql
import boto3
import s3fs
import xarray as xr
from database.connection import openDB, closeDB
from glob import glob
from datetime import datetime

def insertTCs():
    """
    This function will insert tropical cyclone mask files from an S3 bucket into the MySQL database
    """
    db, cur = openDB()
    session = boto3.Session()
    s3 = session.resource('s3')
    bucket = s3.Bucket('tos2ca-dev1')
    files = []
    for objects in bucket.objects.filter(Prefix="tc-quadrant-masks/"):
        files.append(objects.key)
    files = files[1:]
    
    fs_s3 = s3fs.S3FileSystem(anon=False)
    for thisFile in files:
        with fs_s3.open('s3://tos2ca-dev1/' + thisFile, mode='rb') as s3_file_obj:
            ds = xr.open_dataset(s3_file_obj)
            name = ds.storm_name
            startDate = datetime.strptime(ds.start_date, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            endDate = datetime.strptime(ds.end_date, '%Y%m%d%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
            sid = ds.storm_id
            if ds.threshold == 34:
                thirtyFourKnots = 's3://tos2ca-dev1/' + thisFile
            else:
                thirtyFourKnots = None
            if ds.threshold == 50:
                fiftyKnots = 's3://tos2ca-dev1/' + thisFile
            else:
                fiftyKnots = None
            if ds.threshold == 64:
                sixtyFourKnots = 's3://tos2ca-dev1/' + thisFile
            else:
                sixtyFourKnots = None
            
            sql = 'INSERT INTO tropicalCyclones SET name=%s, startDate=%s, endDate=%s, sid=%s, thirtyFourKnots=%s, fiftyKnots=%s, sixtyFourKnots=%s'
            try:
                cur.execute(sql, (name, startDate, endDate, sid, thirtyFourKnots, fiftyKnots, sixtyFourKnots))
                db.commit()
            except pymysql.err.IntegrityError:
                if thirtyFourKnots != None:
                    sql = 'UPDATE tropicalCyclones SET thirtyFourKnots=%s WHERE sid=%s'
                    cur.execute(sql, (thirtyFourKnots, sid))
                    db.commit()
                elif fiftyKnots != None:
                    sql = 'UPDATE tropicalCyclones SET fiftyKnots=%s WHERE sid=%s'
                    cur.execute(sql, (fiftyKnots, sid))
                    db.commit()
                elif sixtyFourKnots != None:
                    sql = 'UPDATE tropicalCyclones SET sixtyFourKnots=%s WHERE sid=%s'
                    cur.execute(sql, (sixtyFourKnots, sid))
                    db.commit()
                else:
                    print('Problem with %s' % sid)
                
    db.close()

    return
