import torch
import torch.nn as nn
import torch.nn.functional as F

class CDN(nn.Module):
    def __init__(self,  dim_in, dim_out):
        super(CDN, self).__init__()
        #self.H = H
        self.fc1 = nn.Linear(dim_in, dim_in, bias=True)
        self.fc2 = nn.Linear(dim_in, dim_out, bias=True)
        self.m = nn.Dropout(0.003)


    def forward(self, X):
        '''
        计算fcn
        '''
        #X = F.leaky_relu(self.fc1(X))#首层全连接
        #X = F.leaky_relu(self.fc2(X))#次层全连接
        #X=self.m(X)#dropout
        X1=F.softmax(X, dim=1)
        return X1,X