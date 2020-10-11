import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

class ETR_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C):
        n,k=list(C.size())
        IC=torch.ones(1, n)
        IV=torch.ones(n, 1)
        V=IC.mm(C)#######V[0,j]=vj
        l2V=torch.log2(V)
        V1=IV.mm(V)
        V2=torch.pow(V1,-1)#################################担心越界

        d = torch.sum(A, dim=1)
        D = torch.diag(d)
        F1=D.mm(C)
        D_V=F1.mul(V2)####D_V[i,j]=di/vj
        l2D_V=torch.log2(D_V)
        aim_DV=D_V.mul(l2D_V)
        hgx=-1.0*IC.mm(aim_DV)####hgx[0,j]=H(G|Xj )j

        F2=0.5*A.mm(C)
        G=IC.mm(F2)#######G[0,j]=Gj

        First_part=V.mul(hgx)
        Second_part=G.mul(l2V)
        third_part=G*(log(n,2))
        total=torch.sum((First_part+Second_part+third_part))
        return total