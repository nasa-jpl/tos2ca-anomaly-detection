import pymysql
from valkey import ValkeyCluster
from utils import tos2ca_secrets


def openDB():
    """
    This will open a connection to the MySQL database
    Pulls authentication info from AWS Secrets Manager
    :return db: A class with a pymysql Connection
    :type db: class 'pymysql.connections.Connection'
    :return cur: A class with a pymysql Cursor
    :type cur: class 'pymysql.cursors.Cursor'
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    if secret:
        user   = secret.get("username")
        passwd = secret.get("password")
        host   = secret.get("host")
        db     = secret.get("db")
    else:
        print("Failed to retrieve database login info from secrets")

    db = pymysql.connect(host=host,
                         user=user,
                         passwd=passwd,
                         db=db)
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
    :return r: A class with a Valkey connection
    :type r: class valkey.client.Valkey
    """
    secret = tos2ca_secrets.get_secret("mysql-tos2causer-tos2ca", "us-west-2")
    if secret:
        host = secret.get("rhost")
        port = secret.get("rport")
        db   = secret.get("rdb")
    else:
        print("Failed to retrieve Elasticache login info from secrets")
    r = ValkeyCluster(host=host,
              port=port,
              ssl=True,
              socket_timeout=1200)
    return r
