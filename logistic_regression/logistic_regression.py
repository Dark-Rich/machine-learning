# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
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
    dataMat,labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = np.arange(-3.0,3.0,0.1)
    y = (-weightsA[0]-weightsA[1]*x)/weightsA[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataMat,labelMat = loadDataSet()
    plotBestFit(stoGradAscent1(dataMat,labelMat))