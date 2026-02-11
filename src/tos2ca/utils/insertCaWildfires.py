import boto3
import s3fs
import xarray as xr
from database.connection import openDB, closeDB
from datetime import datetime
from utils import tos2ca_secrets

def insertWFs():
    """
    This function will insert CA Wildfire mask files in an S3 bucket into a MySQL database
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    bucketName = secret.get("bucket")
    db, cur = openDB()
    session = boto3.Session()
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucketName)
    files = []
    for objects in bucket.objects.filter(Prefix="california-wildfire-masks/"):
        files.append(objects.key)
    files = files[1:]

    fs_s3 = s3fs.S3FileSystem(anon=False)
    for thisFile in files:
        if thisFile.endswith('.nc4'):
            print(thisFile)
            with fs_s3.open('s3://' + bucketName + '/' + thisFile, mode='rb') as s3_file_obj:
                ds = xr.open_dataset(s3_file_obj)
                try:
                    startDate = datetime.strptime(ds.ALARM_DATE, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    startDate = None
                try:
                    endDate = datetime.strptime(ds.CONT_DATE, '%a, %d %b %Y %H:%M:%S %Z').strftime('%Y-%m-%d %H:%M:%S')
                except ValueError:
                    endDate = None
                name = ds.FIRE_NAME.replace('"', '')
                sid = ds.OBJECTID
                hierarchyFile = thisFile.replace('Mask-Output.nc4', 'Mask-Output-Hierarchy.json')
                tocFile = thisFile.replace('Mask-Output.nc4', 'TOC.json')
                sql = 'INSERT INTO caWildfires SET name=%s, startDate=%s, endDate=%s, sid=%s'
                cur.execute(sql, (name, startDate, endDate, sid))    
                sql = 'INSERT INTO predefinedOutput SET sid=%s, startDateTime=%s, phenomenaType="CA Wildfire", location=%s, type="masks"'
                cur.execute(sql, (sid, startDate, 's3://' + bucketName + '/' + thisFile))
                sql = 'INSERT INTO predefinedOutput SET sid=%s, startDateTime=%s, phenomenaType="CA Wildfire", location=%s, type="hierarchy"'
                cur.execute(sql, (sid, startDate, 's3://' + bucketName + '/' + hierarchyFile))
                sql = 'INSERT INTO predefinedOutput SET sid=%s, startDateTime=%s, phenomenaType="CA Wildfire", location=%s, type="toc"'
                cur.execute(sql, (sid, startDate, 's3://' + bucketName + '/' + tocFile))
                db.commit()
    closeDB(db)

    return
