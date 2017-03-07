# -*- coding: utf-8 -*-

import numpy as np

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]) #设置X_0为1.0，theta_0 * X_0就是theta_0
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))
    
#批梯度上升
def gradAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights) #每次用全部的训练样本对参数进行更新
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

#随机梯度上升
def stoGradAscent0(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):  #每次只选取训练样本中的一个对参数进行更新
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * np.dot(error,dataMatrix[i])
    return weights

#随机梯度上升优化
def stoGradAscent1(dataMatrix,classLabels,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0+j+i)+0.01 #alpha会更新，减少振荡
            randIndex = int(np.random.uniform(0,len(dataIndex))) #每次随机取样本，减少周期性的波动
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * np.dot(error,dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights

def plotBestFit(weights):
    #weightsA = weights.getA() #将矩阵作为数组返回，矩阵特有的getA()方法
    weightsA = weights
    x = np.array([1.0,0.5,8])
    y = sigmoid(sum(x*weightsA))
    return y

def main():
    dataMat,labelMat = loadDataSet()
    print plotBestFit(stoGradAscent1(dataMat,labelMat))
    
if __name__ == '__main__':
    main()