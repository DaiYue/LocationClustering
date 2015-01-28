import json
import math
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt
import time

# params

samplingInteval = 10000  # 10 seconds
maxClusterRadius = 0.00125  #
minValidClusterSize = 30  # 30 * 10 = 300 seconds

timeRanges = [[22, 23, 0, 1, 2, 3, 4, 5, 6, 7], [9, 10, 11, 14, 15, 16, 17]]
tagOfTimeRanges = ["home", "office"]

# data structure


class LocationAndTime:
    """Data including location and record time"""

    def __init__(self, _time, _latitude, _longitude):
        self.time = _time
        self.latitude = _latitude
        self.longitude = _longitude

    # def toString(self):
    #     return ("{ time : %d latitude : %s longitude : %s }" % (self.time, self.latitude, self.longitude))

    def time(self):
        return self.time

    def latitude(self):
        return self.latitude

    def longitude(self):
        return self.longitude

class LocationWithTags:
    """Location from cluster result and tags by analysing timestamps"""

    def __init__(self, _latitude, _longitude):
        self.latitude = _latitude
        self.longitude = _longitude
        self.tags = []

    def addTag(self, tag):
        self.tags.append(tag)

# parse json data
file = open("testLocation.json")
jsonArray = json.load(file)["results"]
file.close()

rawDataArray = []
for jsonRecord in jsonArray:
    rawDataArray.append(LocationAndTime(jsonRecord["time"], jsonRecord["lat"], jsonRecord["lon"]))

print("%d records" % len(rawDataArray))

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

print("%d standardized records" % len(dataArray))

# clustering

positionArray = []
for data in dataArray:
    positionArray.append([data.latitude, data.longitude])

distanceMatrix = sch.distance.pdist(positionArray)

linkageMatrix = sch.linkage(positionArray, method='centroid', metric='euclidean')
# Z矩阵：第 i 次循环是第 i 行，这一次[0][1]合并了，它们的距离是[2]，这个类簇大小为[3]

clusterResult = sch.fcluster(linkageMatrix, maxClusterRadius, 'distance')

print("%d clusters" % clusterResult.max())

# filter clusters

allCluster = [[] for row in range(clusterResult.max())]

i = 0
while i < len(clusterResult):
    index = clusterResult[i] - 1
    allCluster[index].append(dataArray[i])
    i += 1

validCluster = []
for cluster in allCluster:
    if (len(cluster) >= 30):
        validCluster.append(cluster)

print("%d valid clusters" % len(validCluster))

# add time tag

results = []
for cluster in validCluster:
    dataInRangeCount = [0] * len(timeRanges)

    sumLa = 0
    sumLo = 0
    count = 0

    for data in cluster:
        sumLa += data.latitude
        sumLo += data.longitude
        count += 1

    avgLa = sumLa / count
    avgLo = sumLo / count

    result = LocationWithTags(avgLa, avgLo)

    for data in cluster:
        hour = time.localtime(data.time)
        i = 0
        while i < len(timeRanges):
            if hour in timeRanges[i]:
                dataInRangeCount[i] += 1
            i += 1

    i = 0
    while i < len(dataInRangeCount):
        if (dataInRangeCount[i] > minValidClusterSize * 0.8):
            result.addTag(tagOfTimeRanges[i])
        i += 1

    results.append(result)

# output are in results

# visualize cluster result

def drawCluster(cluster, color):
    sumLa = 0
    sumLo = 0
    count = 0

    for data in cluster:
        sumLa += data.latitude
        sumLo += data.longitude
        count += 1

    avgLa = sumLa / count
    avgLo = sumLo / count

    maxDist = 0
    for data in cluster:
        maxDist = max(maxDist, math.sqrt((data.latitude - avgLa) ** 2 + (data.longitude - avgLo) ** 2))

    circle = plt.Circle((avgLa, avgLo), maxDist, color=color)
    plt.gca().add_artist(circle)


for position in positionArray:
    plt.plot(position[0], position[1], marker='+')

for cluster in allCluster:
    drawCluster(cluster, 'b')

for cluster in validCluster:
    drawCluster(cluster, 'r')

plt.show()