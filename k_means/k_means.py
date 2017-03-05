# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(filename):
	dataSet = []
	fr = open(filename)
	for line in fr.readlines():
		curLine = line.strip().split('\t')
		fltLine = map(float,curLine)
		dataSet.append(fltLine)
	dataSet = array(dataSet)
	return dataSet

def distEclud(vecA,vecB):
	print vecA,vecB
	return sum(power(vecA - vecB,2))

def randCent(dataSet,k):
	n = shape(dataSet)[1]
	centroids = zeros((k,n))
	for j in range(n):
		minJ = min(dataSet[:,j])
		rangeJ = float(max(dataSet[:,j]) - minJ)
		centroids[:,j] = minJ + rangeJ + random.rand(k)
	return centroids

def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
	m = shape(dataSet)[0]
	clusterAssment = zeros((m,2)) #点属于某个簇
	centroids = createCent(dataSet,k)
	clusterChanged = True
	count = 0
	while clusterChanged:
		clusterChanged = False
		for i in range(m):
			minDist = inf
			minIndex = -1
			for j in range(k):
				distJI = distMeas(centroids[j,:],dataSet[i,:])
				if distJI < minDist:
					minDist = distJI
					minIndex = j
			if clusterAssment[i,0] != minIndex:
				clusterChanged = True
			clusterAssment[i,:] = minIndex,minDist
		for cent in range(k): #更新质心的位置
			ptsInClust = dataSet[nonzero(clusterAssment[:,0]==cent)[0]] #获得每个簇的所有数据
			centroids[cent,:] = mean(ptsInClust,axis=0) #每个簇的平均值设为质心
		count += 1
	return centroids,clusterAssment
def main():
    dataSet = loadDataSet('data.txt')
    kMeans(dataSet,3)

if __name__ == '__main__':
	main()