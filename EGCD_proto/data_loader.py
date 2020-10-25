# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
# 导入数据：分隔符为空格
from igraph  import *

def load_cora():
    raw_data = pd.read_csv('data/cora/cora.content',sep = '\t',header = None)
    num = raw_data.shape[0] # 样本点数2708
    # 将论文的编号转[0,2707]
    a = list(raw_data.index)
    b = list(raw_data[0])
    c = zip(b,a)
    map = dict(c)
    # 将词向量提取为特征,第二行到倒数第二行
    features =raw_data.iloc[:,1:-1]
     # 检查特征：共1433个特征，2708个样本点
    #print(features.shape)
    labels = pd.get_dummies(raw_data[1434])
    #print(labels.shape)

    #导入论文引用数据
    raw_data_cites = pd.read_csv('data/cora/cora.cites',sep = '\t',header = None)
    # 创建一个规模和邻接矩阵一样大小的矩阵
    Amatrix = np.zeros((num,num))
    # 创建邻接矩阵
    for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
        x = map[i] ; y = map[j]  #替换论文编号为[0,2707]
        Amatrix[x][y] = Amatrix[y][x] = 1 #有引用关系的样本点之间取1
    # 查看邻接矩阵的元素和（按每列汇总）
    #print((Amatrix.shape))

    return features,Amatrix,labels

def load_dblp():
    features = np.load('data/dblp/dblp_medium_features.npy')
    print(features.shape)
    labels = np.load('data/dblp/dblp_medium_label.npy')
    print('+++++++++++++++++++')
    print(labels.shape)
    adj_data = np.load('data/dblp/dblp_medium_adj.npz')
    indices=adj_data['indices']
    indptr=adj_data['indptr']
    format=adj_data['format']
    shape = adj_data['shape']
    data = adj_data['data']
    print('+++++++++++++++++++')
    print(indices.shape )
    print(indptr.shape)
    print(format.shape)
    print(shape.shape)
    print(data.shape)
    print('+++++++++++++++++++')
    print(indices)
    print('+++++++++++++++++++')
    print(indptr)
    print('+++++++++++++++++++')
    print(format)
    print('+++++++++++++++++++')
    print(shape)
    print(data)
def load_football():
    path='data/football/football.gml'
    g = Graph.Read_GML(path)
    Amatrix = g.get_adjacency()
    #L = g.get_label()
    Amatrix = np.array(Amatrix.data)
    #L = np.array(L.data)
    #print(Amatrix)
    lambda1, features = np.linalg.eig(Amatrix)
    labels_temp=(g.vs['value'])
    #plot(g)

    [n,d]=features.shape
    labels=np.zeros((n,12))
    for i in range(n):
        for j in range(12):
            if(labels_temp[i]==j):
                labels[i,j]=1.0
    print(features.shape)
    return features, Amatrix, labels
def load_six():
    features=np.eye(6)
    Amatrix=[
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
             ]
    Amatrix=np.mat(Amatrix)
    labels=[[0,0,0,1,1,1],
            [1,1,1,0,0,0]]
    labels=np.array(labels).T
    return features, Amatrix, labels
#print(load_football())

