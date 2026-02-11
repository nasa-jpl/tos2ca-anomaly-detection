import blosc
import pickle as pkl

from database.connection import openCache


def serialize_dict(data):
    """
    Function to change the data dict to bytes and
    serialize it.
    :param data: the data to convert
    :type data: dict
    :return: the dict as serialized bytes
    :rtype: bytes
    """
    return pkl.dumps(data, protocol=pkl.HIGHEST_PROTOCOL)


def split_bytes(data):
    """
    Function to split data in to chunks.
    :param data: the data to chunk
    :type data: bytes
    :return: list of data chunks
    :rtype: list
    """
    chunk_size = 100 * 1024 * 1024
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def setData(data, jobInfo, start_time, jobID, chunkID):
    """
    Function to store the data and jobInfo variables in Elasticache.  It 
    turns the dict objects into bytes first, so that there are no TypeErrors
    when loading the data into Redis. Automatically sets expiry time for each
    entry to 1 day.
    :param data: data from a reader
    :type data: dict
    :param jobInfo: information about the job
    :type jobInfo: dict
    :param start_time: start time of this job
    :type start_time: datetime
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    """
    r = openCache()
    key = 'job%s-%s-data' % (jobID, chunkID)
    serialized = serialize_dict(data)
    chunks = split_bytes(serialized)
    for idx, chunk in enumerate(chunks):
        compressedChunk = blosc.compress(chunk)
        r.set(f"{key}:{idx}", compressedChunk)
    r.set(f"{key}:chunk_count", len(chunks))
    jobInfoToBytes = pkl.dumps(jobInfo)
    r = openCache()
    r.set('job%s-%s-jobInfo' % (jobID, chunkID), jobInfoToBytes, ex=86400)
    startTimeToBytes = pkl.dumps(start_time)
    r = openCache()
    r.set('job%s-%s-start_time' % (jobID, chunkID), startTimeToBytes, ex=86400)
    print('Insert to Elasticache complete')

    return


def setFortraccData(fortraccData, jobID, chunkID):
    """
    Function to store FortraCC data in Elasticache.  It 
    turns the dict objects into bytes first, so that there are no TypeErrors
    when loading the data into Redis. Automatically sets expiry time for each
    entry to 1 day.
    :param fortraccData: processed sparse FortraCC data
    :type fortraccData: fortracc_module.flow.SparseTimeOrderedSequence
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    """
    dataToBytes = pkl.dumps(fortraccData)
    compressedDataToBytes = blosc.compress(dataToBytes)
    r = openCache()
    r.set('fortracc-%s-%s' % (jobID, chunkID), compressedDataToBytes, ex=86400)
    print('Insert to Elasticache complete')

    return


def setAuxGeoIRData(auxgeoirData, jobID, chunkID):
    """
    Function to store AuxGEOIR data in Elasticache.  It 
    turns the dict objects into bytes first, so that there are no TypeErrors
    when loading the data into Redis. Automatically sets expiry time for each
    entry to 1 day.
    :param fortraccData: processed sparse AuxGeoIR data
    :type fortraccData: fortracc_module.flow.SparseTimeOrderedSequence
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    """
    dataToBytes = pkl.dumps(auxgeoirData)
    compressedDataToBytes = blosc.compress(dataToBytes)
    r = openCache()
    r.set('auxgeoir-%s-%s' % (jobID, chunkID), compressedDataToBytes, ex=86400)
    print('Insert to Elasticache complete')

    return


def getData(jobID, chunkID):
    """
    Function to retrieve the data and jobInfo variables in Elasticache.  It 
    converts the variables back from bytes to dicts when loading the data out of Redis.
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    :return data: data from a reader
    :type data: dict
    :return jobInfo: information about the job
    :type jobInfo: dict
    :return start_time: start time of this job
    :type jobInfo: datetime
    """
    r = openCache()
    key = 'job%s-%s-data' % (jobID, chunkID)
    chunkCount = int(r.get(f"{key}:chunk_count"))
    chunks = []
    for i in range(chunkCount):
        compressedDataBack = r.get(f"{key}:{i}")
        chunkBack = blosc.decompress(compressedDataBack)
        chunks.append(chunkBack)
    serialized = b''.join(chunks)
    data = pkl.loads(serialized)
    r = openCache()
    jobInfoBack = r.get('job%s-%s-jobInfo' % (jobID, chunkID))
    jobInfo = pkl.loads(jobInfoBack)
    r = openCache()
    startTimeBack = r.get('job%s-%s-start_time' % (jobID, chunkID))
    start_time = pkl.loads(startTimeBack)

    return (data, jobInfo, start_time)


def getFortraccData(jobID, chunkID):
    """
    Function to retrieve the FortraCC data from Elasticache.  It 
    converts the variables back from bytes to dicts when loading the data out of Redis.
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    :return fortraccData: data from FortraCC
    :type fortraccData: fortracc_module.flow.SparseTimeOrderedSequence
    """
    r = openCache()
    compressedDataBack = r.get('fortracc-%s-%s' % (jobID, chunkID))
    dataBack = blosc.decompress(compressedDataBack)
    fortraccData = pkl.loads(dataBack)
    
    return fortraccData


def getAuxGeoIRData(jobID, chunkID):
    """
    Function to retrieve the AuxGeoIR data from Elasticache.  It 
    converts the variables back from bytes to dicts when loading the data out of Redis.
    :param jobID: the Job ID #
    :type jobID: int
    :param chunkID: the Chunk ID #
    :type chunkID: int
    :return fortraccData: data from AuxGeoIR
    :type fortraccData: fortracc_module.flow.SparseTimeOrderedSequence
    """
    r = openCache()
    compressedDataBack = r.get('auxgeoir-%s-%s' % (jobID, chunkID))
    dataBack = blosc.decompress(compressedDataBack)
    getAuxGeoIRData = pkl.loads(dataBack)
    
    return getAuxGeoIRData
