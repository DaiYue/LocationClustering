import json
import math
import scipy.cluster.hierarchy as sch
import time

# params

defaultTimeRanges = [[22, 23, 0, 1, 2, 3, 4, 5, 6, 7], [9, 10, 11, 14, 15, 16, 17]]
defaultTagOfTimeRanges = ["home", "office"]
version = '0.1'

# data structure

class LocationAndTime:
    """Data including location and record time"""

    def __init__(self, _time, _latitude, _longitude):
        self.time = _time
        self.latitude = _latitude
        self.longitude = _longitude

    def time(self):
        return self.time

    def latitude(self):
        return self.latitude

    def longitude(self):
        return self.longitude

class LocationWithTags:
    """Location from cluster result and tags by analysing timestamps"""

    def __init__(self, _latitude, _longitude, _tags):
        self.latitude = _latitude
        self.longitude = _longitude
        self.tags = _tags
        self.estimateTime = 0

class TagInfo:
    def __init__(self, _tag, _estimateTime, _ratio):
        self.tag = _tag
        self.estimateTime = _estimateTime
        self.ratio = _ratio

def cluster(jsonArray, maxClusterRadius=0.00125, samplingInteval=10000,
                       timeRanges=defaultTimeRanges, tagOfTimeRanges=defaultTagOfTimeRanges, timeThreshold = 300000,
                       ratioThreshold = 0.4):
    # parse json data

    rawDataArray = []
    for jsonRecord in jsonArray:
        rawDataArray.append(LocationAndTime(jsonRecord["time"], jsonRecord["lat"], jsonRecord["lon"]))

    # print("%d records" % len(rawDataArray))

    # sampling by time

    rawDataArray.sort(key=LocationAndTime.time)

    dataArray = []
    i = 0
    while i < len(rawDataArray):
        floor = math.floor(rawDataArray[i].time / samplingInteval)
        bottom = floor * samplingInteval
        top = (floor + 1) * samplingInteval

        latitudeSum = 0
        longitudeSum = 0
        count = 0
        while i < len(rawDataArray) and rawDataArray[i].time in range(bottom, top):
            latitudeSum += rawDataArray[i].latitude
            longitudeSum += rawDataArray[i].longitude
            count += 1
            i += 1

        dataArray.append(LocationAndTime(bottom, latitudeSum / count, longitudeSum / count))

    # print("%d standardized records" % len(dataArray))

    # clustering

    positionArray = []
    for data in dataArray:
        positionArray.append([data.latitude, data.longitude])

    distanceMatrix = sch.distance.pdist(positionArray)

    linkageMatrix = sch.linkage(positionArray, method='centroid', metric='euclidean')
    # Z矩阵：第 i 次循环是第 i 行，这一次[0][1]合并了，它们的距离是[2]，这个类簇大小为[3]

    clusterResult = sch.fcluster(linkageMatrix, maxClusterRadius, 'distance')

    # print("%d clusters" % clusterResult.max())

    # filter clusters

    allCluster = [[] for row in range(clusterResult.max())]

    i = 0
    while i < len(clusterResult):
        index = clusterResult[i] - 1
        allCluster[index].append(dataArray[i])
        i += 1

    validCluster = []
    for cluster in allCluster:
        if (len(cluster) >= timeThreshold / samplingInteval):
            validCluster.append(cluster)

    # print("%d valid clusters" % len(validCluster))

    # add tags

    globalDataInRangeCount = countDataInRange(dataArray, timeRanges)

    results = []
    for cluster in validCluster:
        clusterDataInRangeCount = countDataInRange(cluster, timeRanges)

        tags = []
        i = 0
        while i < len(timeRanges):
            if globalDataInRangeCount[i] == 0:
                i += 1
                continue
            ratio = clusterDataInRangeCount[i] / globalDataInRangeCount[i]
            if ratio > ratioThreshold:
                estimateTime = clusterDataInRangeCount[i] * samplingInteval
                tags.append(TagInfo(tagOfTimeRanges[i], estimateTime, ratio))
            i += 1

        if len(tags) == 0:  # no tag
            continue

        sumLa = 0
        sumLo = 0
        count = 0

        for data in cluster:
            sumLa += data.latitude
            sumLo += data.longitude
            count += 1

        avgLa = sumLa / count
        avgLo = sumLo / count

        result = LocationWithTags(avgLa, avgLo, tags)
        result.estimateTime = count * samplingInteval

        results.append(result)

    # output are in results
    return json.dumps(results,default=lambda obj:obj.__dict__)

def countDataInRange(dataArray, timeRanges):
    dataInRangeCount = [0] * len(timeRanges)
    for data in dataArray:
        timeStamp = time.localtime(data.time / 1000)
        i = 0
        while i < len(timeRanges):
            if timeStamp.tm_hour in timeRanges[i]:
                dataInRangeCount[i] += 1
            i += 1
    return dataInRangeCount