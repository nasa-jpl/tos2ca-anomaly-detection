import smtplib
from database.connection import openDB, closeDB
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def phdefComplete(jobID):
    """
    Sends email notification to user letting them know
    their PhDef job is complete and where to get their 
    files in S3.
    :param jobID: ID number of the PhDef job
    :type jobID: int
    """
    db, cur = openDB()
    sql = 'SELECT u.email FROM users u, jobs j WHERE j.jobID=%s AND u.userID=j.userID'
    cur.execute(sql, (jobID))
    userInfo = cur.fetchall()
    sql = 'SELECT location FROM output WHERE type="masks output" AND jobID=%s LIMIT 1'
    cur.execute(sql, (jobID))
    results = cur.fetchall()
    print(results)
    bucketName = results[0]['location'].split('/')[2]

    server = smtplib.SMTP('localhost', 25)
    server.ehlo()

    fromaddr = 'tos2ca-noreply@jpl.nasa.gov'
    toaddr = [userInfo[0]['email']]
    subject = 'TOS2CA PhDef Job #%s Complete' % jobID

    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = ', '.join(toaddr)
    msg['Subject'] = subject
    body = 'Your TOS2CA Phenomena Definition Job #%s has completed\n\n' % jobID
    body += 'You may pick up your job files in s3://%s/%s\n\n' % (
        bucketName, jobID)
    body += 'To look up information on your past jobs, please visit the <a href="https://tos2ca-dev1.jpl.nasa.gov/job_lookup.php">TOS2CA Job Lookup</a> page.\n\n'
    body += 'Sincerely,\n\n'
    body += 'The TOS2CA Team'
    msg.attach(MIMEText(body, 'html', 'utf-8'))
    text = msg.as_string()
    try:
        server.sendmail(fromaddr, toaddr, text)
        server.close
        print('Message sent')
    except:
        print('Message failed to send.')

    closeDB(db)

    return

def curationComplete(jobID):
    """
    Sends email notification to user letting them know
    their Curation job is complete and where to get their 
    files in S3.
    :param jobID: ID number of the curation job
    :type jobID: int
    """
    db, cur = openDB()
    sql = 'SELECT u.email FROM users u, jobs j WHERE j.jobID=%s AND u.userID=j.userID'
    cur.execute(sql, (jobID))
    userInfo = cur.fetchall()
    sql = 'SELECT location FROM output WHERE type="curated subset" AND jobID=%s LIMIT 1'
    cur.execute(sql, (jobID))
    results = cur.fetchall()
    print(results)
    bucketName = results[0]['location'].split('/')[2]

    server = smtplib.SMTP('localhost', 25)
    server.ehlo()

    fromaddr = 'tos2ca-noreply@jpl.nasa.gov'
    toaddr = [userInfo[0]['email']]
    subject = 'TOS2CA Curation Job #%s Complete' % jobID

    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = ', '.join(toaddr)
    msg['Subject'] = subject
    body = 'Your TOS2CA Data Curation Job #%s has completed\n\n' % jobID
    body += 'You may pick up your job files in s3://%s/%s\n\n' % (
        bucketName, jobID)
    body += 'To look up information on your past jobs, please visit the <a href="https://tos2ca-dev1.jpl.nasa.gov/job_lookup.php">TOS2CA Job Lookup</a> page.\n\n'
    body += 'Sincerely,\n\n'
    body += 'The TOS2CA Team'
    msg.attach(MIMEText(body, 'html', 'utf-8'))
    text = msg.as_string()
    try:
        server.sendmail(fromaddr, toaddr, text)
        server.close
        print('Message sent')
    except:
        print('Message failed to send.')

    closeDB(db)

    return