import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import numpy as np

class ETR_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A,B, C,gama):
        n,k=list(C.size())
        IC=torch.ones(1, n)
        IV=torch.ones(n, 1)
        V=IC.mm(C)#######V[0,j]=vj
        l2V=torch.log2(V+1e-20)
        V1=IV.mm(V)
        V2=torch.pow(V1+1e-20,-1)#################################担心越界

        d = torch.sum(A, dim=1)
        D = torch.diag(d)
        F1=D.mm(C)
        D_V=F1.mul(V2)####D_V[i,j]=di/vj
        l2D_V=torch.log2(D_V+1e-20)
        aim_DV=D_V.mul(l2D_V)
        hgx=-1.0*IC.mm(aim_DV)####hgx[0,j]=H(G|Xj )j

        F2=0.5*A.mm(C)
        G=IC.mm(F2)#######G[0,j]=Gj

        m=torch.sum(A)
        First_part=V.mul(hgx)
        Second_part=G.mul(l2V)
        third_part=G*(log(2*m,2))

        constraint=torch.norm((k/n)*(B.t().mm(B))-torch.eye(k))
        total=torch.sum((First_part-Second_part+third_part))+gama*(constraint)
        return total

class ETR_loss_diag(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C ):
        n, k = list(C.size())
        IsumR = torch.ones(n, 1)
        p=torch.diag(C.t().mm(A.mm(C)))
        kk=torch.log2((C.t().mm(A)).mm(IsumR)+1e-20)
        total=torch.sum((torch.diag(C.t().mm(A.mm(C)))).mul((torch.log2((C.t().mm(A)).mm(IsumR)+1e-20)).t()))
        print(total)
        return total

class ETR_loss_trace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C ):
        n, k = list(C.size())
        IsumC = torch.ones(1,n)
        IsumCTC=torch.ones(1,k)
        IsumCDC = torch.ones(k, 1)

        total=torch.trace((1/(torch.sum(A)))*(C.t().mm(A.mm(C))).mul(torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C)))*(1/(torch.sum(A))))+1e-40)))
        constraint = 0.1*(torch.norm( (C.t().mm(C)) - torch.eye(k)))
        #C1 = torch.pow(IsumCDC.mm(IsumCTC.mm((C.t().mm(C)))) + 1e-20, -1)
        #constraint_1 = torch.norm((C.mm(C.t())))
        #print(total)
        total=total# + constraint
        return total

'''
from sklearn import metrics
labels=[1,1,2]
labels_true=[1,1,2]
print(metrics.normalized_mutual_info_score(labels,labels_true))
labels=[2,2,1]
labels_true=[1,1,2]
print(metrics.normalized_mutual_info_score(labels,labels_true))
labels=[1,1,2,3,4,4,4]
labels_true=[1,1,2,3,5,4,4]
print(metrics.normalized_mutual_info_score(labels,labels_true))
labels=[3,3,1,2,5,5,5]
labels_true=[1,1,2,3,5,4,4]
print(metrics.normalized_mutual_info_score(labels,labels_true))
labels=[3,3,1,2,5,5,4]
labels_true=[1,1,2,3,5,4,4]
print(metrics.normalized_mutual_info_score(labels,labels_true))
import numpy as np
A=np.eye(4)
X=torch.Tensor(A)
print(torch.sum(X))

'''
'''
Amatrix = [
    [0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0],
    [1, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0],
]
Amatrix = np.mat(Amatrix)
A=torch.Tensor(Amatrix)
labels=[[0,0,0,1,1,1],
            [1,1,1,0,0,0]]
labels=np.array(labels).T
C=torch.Tensor(labels)
n, k = list(C.size())
IsumC = torch.ones(1,n)
IsumCTC=torch.ones(1,k)
IsumCDC = torch.ones(k, 1)

total = torch.trace(
    (1 / (torch.sum(A))) * (C.t().mm(A.mm(C))).mul(torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C))) * (1 / (torch.sum(A)))))))
print(total)
'''
