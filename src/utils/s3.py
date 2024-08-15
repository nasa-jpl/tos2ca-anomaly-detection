import boto3
import logging
import os
import pymysql
import requests
from datetime import datetime, timedelta
from botocore.exceptions import ClientError


def s3Upload(jobID, info, bucketName, db, cur):
    """
    Utility to write files to the TOS2CA S3 bucket
    Must have your AWS credentials in ~/.aws/config for the profile 
    you are trying to authenticate with
    :param jobID:  ID number of the job
    :type jobID: int
    :param info: dictionary containing the filename with local path (string), 
                 startdateTime (string as "YYYY-MM-DD HH:MM:SS"), and type (string)
    :type info: dict
    :param bucketName: name of the S3 bucket
    :type bucketName: string
    :param db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    """
    session = boto3.Session()
    s3Client = session.client('s3')
    try:
        response = s3Client.upload_file(
            info['filename'], bucketName, f"{jobID}/{info['filename'].split('/')[-1]}")
        bucketLocation = 's3://%s/%s/%s' % (bucketName,
                                            jobID, info['filename'].split('/')[-1])
        sql = 'INSERT INTO output SET jobID=%s, location=%s, type=%s, startDateTime=%s'
        try:
            cur.execute(sql, (jobID, bucketLocation, info['type'], info['startDateTime']))
            db.commit()
        except pymysql.err.IntegrityError:
            logging.info('File already in database output table.')
        logging.info(response)
        os.remove(info['filename'])
        logging.info('File %s removed locally' % info['filename'])
    except ClientError as e:
        logging.error(e)
        return False
    return True


def s3Delete(jobID, bucketName, db, cur):
    """
    Utility to delete all files from a job in the TOS2CA S3 bucket
    Must have your AWS credentials in ~/.aws/config for the profile 
    you are trying to authenticate with
    :param jobID:  ID number of the job
    :type jobID: int
    :param bucketName: name of the S3 bucket
    :type bucketName: string
    :param db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :param cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    """
    session = boto3.Session()
    s3Client = session.client('s3')
    sql = 'SELECT location FROM output WHERE jobID=%s'
    cur.execute(sql, (jobID))
    fileList = cur.fetchall()
    for thisFile in fileList:
        fileKey = '/'.join(thisFile.split('/')[3:])
        try:
            response = s3Client.delete_object(Bucket=bucketName, Key=fileKey)
            sql = 'DELETE FROM output SET jobID=%s, location=%s'
            cur.execute(sql, (jobID, thisFile))
            db.commit()
            logging.info(response)
        except ClientError as e:
            logging.error(e)
            return False
        return True
    
def s3GetTemporaryCredentials(daac):
    """
    Gets temporary credentials from a DAAC for S3 access.  User must have a .netrc file 
    in their home directory with their NASA Earthdata credentials that has the line:
           machine urs.earthdata.nasa.gov login <USERNAME> password <PASSWORD>
    :param daac: name of the DAAC 
    :type daac: str
    :return s3_creds: temporary credential info
    :type resp: dict
    """
    s3_cred_endpoint = {
        'PO.DAAC':'https://archive.podaac.earthdata.nasa.gov/s3credentials',
        'GES-DISC': 'https://data.gesdisc.earthdata.nasa.gov/s3credentials',
    }
    endpoint = s3_cred_endpoint[daac]
    resp=requests.get(endpoint)
    s3_creds=resp.json()

    return s3_creds

def checkReauth(creds):
    """
    Function to see if NASA Earthdata credentials have expired
    :param creds: NASA Earthdata credentials
    :type creds: dict
    :return newCredsNeeded: indicates if the credentials have expired
    :rtype newCredsNeeded: bool
    """
    expiration = creds['expiration']
    expiration = datetime.strptime(creds['expiration'][:-6], '%Y-%m-%d %H:%M:%S')
    currentUTC = datetime.now() + timedelta(minutes=10)
    if expiration >= currentUTC:
        newCredsNeeded = 0
    else:
        newCredsNeeded = 1

    return newCredsNeeded