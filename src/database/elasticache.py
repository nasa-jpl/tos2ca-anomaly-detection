from pickle import dumps, loads

def setData(r, data, jobInfo, start_time, jobID):
    """
    Function to store the data and jobInfo variables in Elasticache.  It 
    turns the dict objects into bytes first, so that there are no TypeErrors
    when loading the data into Redis.
    :param r: the Elasticache connection
    :type r: class 'redis.client.Redis'
    :param data: data from a reader
    :type data: dict
    :param jobInfo: information about the job
    :type jobInfo: dict
    :param start_time: start time of this job
    :type start_time: datetime
    :param jobID: the Job ID #
    :type jobID: int
    """
    dataToBytes = dumps(data)
    r.set('job%s-data' % jobID, dataToBytes, ex=172800)
    jobInfoToBytes = dumps(jobInfo)
    r.set('job%s-jobInfo' % jobID, jobInfoToBytes, ex=172800)
    startTimeToBytes = dumps(start_time)
    r.set('job%s-start_time' % jobID, startTimeToBytes, ex=172800)
    
    return

def getData(r, jobID):
    """
    Function to retrieve the data and jobInfo variables in Elasticache.  It 
    converts the variables back from bytes to dicts when loading the data out of Redis.
    :param r: the Elasticache connection
    :type r: class 'redis.client.Redis'
    :param jobID: the Job ID #
    :type jobID: int
    :return data: data from a reader
    :type data: dict
    :return jobInfo: information about the job
    :type jobInfo: dict
    :return start_time: start time of this job
    :type jobInfo: datetime
    """
    dataBack = r.get('job%s-data' % jobID)
    jobInfoBack = r.get('job%s-jobInfo' % jobID)
    startTimeBack = r. get('job%s-start_time' % jobID)
    data = loads(dataBack)
    jobInfo = loads(jobInfoBack)
    start_time = loads(startTimeBack)

    return (data, jobInfo, start_time)