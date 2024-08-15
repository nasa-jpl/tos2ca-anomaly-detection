import pymysql
import os
import redis
from utils import tos2ca_secrets

def openDB():
    """
    This will open a connection to the MySQL database
    :return db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :return cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    """
    user   = os.getenv('DBUSERNAME')
    passwd = os.getenv('DBPASSWORD')
    if user is None:
        secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
        if secret:
            user   = secret.get("username")
            passwd = secret.get("password")
            host   = secret.get("host")
        else:
            print("Failed to retrieve database login info from secrets")
    else:
        host = "tos2cadev1.ctznfzbiztp3.us-west-2.rds.amazonaws.com"

    db = pymysql.connect(host=host,
                         user=user,
                         passwd=passwd,
                         db="tos2ca")
    cur = db.cursor(pymysql.cursors.DictCursor)
    return (db, cur)


def closeDB(db):
    """
    This will close a connection to the MySQL database
    :param db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    """
    db.close()

def openCache():
    """
    This will connect to the AWS Elasitcache host
    :return r: A class with a Redis connection
    :type r: class edis.client.Redis
    """
    r = redis.Redis(host="master.tos2ca1.5khw6z.usw2.cache.amazonaws.com", 
                    port=6379, 
                    db=0)
    return r
