# -*- coding: utf-8 -*-

from numpy import *

def replaceNanWithMean():
	dataMat = loadDataSet('data.txt',' ')
	numFeat = shape(dataMat)[1]
	for i in range(numFeat):
		meanVal = mean(dataMat[nonzero(~isnan(dataMat[:,i].A))[0],i]) #非空值的平均数
		dataMat[nonzero(isnan(dataMat[:,i].A))[0],i] = meanVal #空值为平均数
	return dataMat

def loadDataSet(fileName,delim='\t'):
	fr = open(fileName)
	stringArr = [line.strip().split(delim) for line in fr.readlines()]
	datArr = [map(float,line) for line in stringArr]
	return mat(datArr)

def pca(dataMat,topNfeat=9999999):
	meanVal = mean(dataMat,axis=0) #平均值
	meanRemoved = dataMat - meanVal #减去平均值
	covMat = cov(meanRemoved,rowvar=0) #协方差
	eigVals,eigVects = linalg.eig(mat(covMat)) #特征值、特征向量
	eigValInd = argsort(eigVals) #小到大排序
	eigValInd = eigValInd[:-(topNfeat+1):-1] #删除不需要的维度
	redEigVects = eigVects[:,eigValInd] #特征向量从大到小
	lowDataMat = meanRemoved * redEigVects #转换数据到新的维度
	reconMat = (lowDataMat * redEigVects.T) + meanVal
	return lowDataMat,reconMat

if __name__ == '__main__':
	dataSet = replaceNanWithMean()
	pca(dataSet)