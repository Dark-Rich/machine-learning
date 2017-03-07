# -*- coding: utf-8 -*-

from numpy import *
from numpy import linalg as la

def loadDataSet():
    return array([[0, 0, 0, 2, 2],
    	[3, 1, 0, 3, 3],
		[0, 0, 2, 1, 1],
		[1, 1, 1, 0, 0],
		[2, 2, 3, 1, 0],
		[5, 5, 5, 0, 0],
		[1, 1, 1, 0, 0]])

def svd(dataSet,N=2):
	U,sigma,VT = la.svd(dataSet) #svd分解
	sigma2 = mat([[sigma[0],0],[0,sigma[1]]])
	return U[:,:2] * sigma2 * VT[:N,:] #还原数据

def ecludSim(vecA,vecB): #欧式距离
	return 1.0 / (1.0 + la.norm(vecA - vecB))

def cosSim(vecA,vecB): #余弦相似度
	num = float(sum(vecA * vecB))
	denom = la.norm(vecA) * la.norm(vecB)
	return 0.5 + 0.5 * (num / denom)

def perasSim(vecA,vecB): #皮尔森相关性
	if len(vecA) < 3:
		return 1.0
	return 0.5 + 0.5 * corrcoef(vecA,vecB,rowvar=0)[0][1]

def standEst(dataSet, user, simMeas, item): #对物品估计评分
	n = shape(dataSet)[1]
	simTotal = 0.0
	ratSimTotal = 0.0
	for j in range(n):
		userRating = dataSet[user,j]
		if userRating == 0 or j==item:
			continue
		overLap = nonzero(logical_and(dataSet[:,item]>0,dataSet[:,j]>0))[0] #同时大于0 
		if len(overLap) == 0:
			similarity = 0
		else:
			similarity = simMeas(dataSet[overLap,item],dataSet[overLap,j])
		print 'the %d and %d similarity is: %f' % (item, j, similarity)
		simTotal += similarity
		ratSimTotal += similarity * userRating
	if simTotal == 0:
		return 0
	else:
		return ratSimTotal / simTotal

def recommend(dataSet,user,N=2,simMeas=ecludSim,estMethod=standEst):
	unratedItems = nonzero(dataSet[user,:]==0)[0] #发现未评级物品
	if len(unratedItems) == 0:
		return
	itemScores = []
	for item in unratedItems:
		estimatedScore = estMethod(dataSet,user,simMeas,item)
		itemScores.append((item,estimatedScore))
	return sorted(itemScores,key=lambda jj:jj[1],reverse=True)[:N]

def main():
	dataSet = loadDataSet()
	print recommend(dataSet,5)

if __name__ == '__main__':
	main()