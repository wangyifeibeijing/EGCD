import igraph as ig
import numpy as np

#community_infomap(self, edge_weights=None, vertex_weights=None, trials=10)
#edge_weights：边属性的一个名字或一个包含边权值的list
#vertex_weights：节点属性名或包含节点权值的list
#trials：预期将网络分割的数目，默认值为10，并没有对其进行详细说明
#输出 VertexClustering，是一个以列表为元素的列表
def use_infomap(A,cluster_num, weighted=False):
    [n,_]=A.shape
    node = list(range(0, n ))
    L = []
    wei = []
    label = [0 for index in range(n)]
    for i in range(n):
        for j in range(n):
            if(A[i,j]!=0)and (A[j,i]!=0):
                L.append((i, j))
                if(weighted):
                    wei.append(0.5*(A[i,j]+A[j,i]))
                else:
                    wei.append(1)
    g = ig.Graph.TupleList(L, directed=False)  # directed函数将g定义为有向图
    community_list1 = g.community_infomap(edge_weights=wei,trials=cluster_num)

    com_fla=0
    for community in community_list1:
        for i in community:
            label[i]=com_fla
        com_fla=com_fla+1
    return label

if __name__ == '__main__':

    Amatrix=[
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
                 ]
    Amatrix=np.mat(Amatrix)
    C= use_infomap(Amatrix,2)
    print (C)
