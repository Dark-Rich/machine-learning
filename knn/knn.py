# -*- coding: utf-8 -*-

import operator
import numpy as np

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
    
#KNN
def kNN(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #读取第一维大小
    diffMat = np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1) #行向量相加
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort() #从小到大返回索引值
    classCount = {} #字典
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #k个邻居的标签
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1 #返回指定键的值+1，不存在返回0
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    intX = [0,0]
    group,labels = createDataSet()
    print(kNN(intX,group,labels,2))