# -*- coding: utf-8 -*-

from numpy import *
from numpy import linalg as la

def loadDataSet(filename):
	fr = open(filename)
	numFeat = len(fr.readline().split('\t')) - 1
	dataSet = []
	labelSet = []
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		dataSet.append(lineArr)
		labelSet.append(float(curLine[-1]))
	return dataSet,labelSet

def standRegres(dataSet,labelSet):
	xMat = mat(dataSet)
	yMat = mat(labelSet).T
	xTx = dot(xMat.T,xMat) 
	if la.det(xTx) == 0.0:
		return
	W = dot(xTx.I,dot(xMat.T,yMat))
	return W

def main():
	dataSet,labelSet = loadDataSet('data.txt')
	W = standRegres(dataSet,labelSet).reshape(1,2).getA()
	x = array([1,0.58])
	print sum(W * x)

if __name__ == '__main__':
	main()