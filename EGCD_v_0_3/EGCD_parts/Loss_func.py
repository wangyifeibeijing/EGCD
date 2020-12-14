import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import numpy as np
import data_loader



class ETR_loss_trace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C):
        n, k = list(C.size())
        IsumC = torch.ones(1,n)
        IsumCTC=torch.ones(1,k)
        IsumCDC = torch.ones(k, 1)
        total=torch.trace((1/(torch.sum(A)))*(C.t().mm(A.mm(C))).mul(torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C)))*(1/(torch.sum(A))))+1e-40)))
        #constraint = 0.001*(torch.norm( (k/n)*(C.t().mm(C)) - torch.eye(k)))

        #total=total + constraint
        return total
class ETR_loss_H(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C ,H,papra,flag):
        n, k = list(C.size())
        IsumC = torch.ones(1,n)
        IsumCTC=torch.ones(1,k)
        IsumCDC = torch.ones(k, 1)
        total=torch.trace((1/(torch.sum(A)))*(C.t().mm(A.mm(C))).mul(torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C)))*(1/(torch.sum(A))))+1e-40)))
        d = A.sum(1)  # 所有节点的度
        D = torch.diag(d)  # D = D^-1/2
        h_part=torch.trace(H.t().mm(D.mm(H)))
        #constraint = (torch.norm( (k/n)*(C.t().mm(C)) - torch.eye(k)))
        if(flag>10):
            total=total + papra*h_part
        return total,h_part

def use_loss(A, C):
    n, k = list(C.size())
    IsumC = torch.ones(1, n)
    IsumCTC = torch.ones(1, k)
    IsumCDC = torch.ones(k, 1)
    total = torch.trace((1 / (torch.sum(A))) * (C.t().mm(A.mm(C))).mul(
        torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C))) * (1 / (torch.sum(A)))) + 1e-40)))
    constraint = (torch.norm((k / n) * (C.t().mm(C)) - torch.eye(k)))
    return total
if __name__ == '__main__':
    features, Amatrix, labels = data_loader.load_polblogs()
    features = np.array(features.astype("float32"))
    Amatrix = np.array(Amatrix.astype("float32"))
    labels = np.array(labels.astype("float32"))
    A1 = torch.FloatTensor(Amatrix.astype("float32"))  # 邻接矩阵转tensor
    X = torch.Tensor(features)
    C=torch.Tensor(labels)
    print(use_loss(A1,C))
