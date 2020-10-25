import torch
import torch.nn as nn
import torch.nn.functional as F

class CDN(nn.Module):
    def __init__(self,  dim_in, dim_out):
        super(CDN, self).__init__()
        #self.H = H
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_out, bias=False)
        self.m = nn.Dropout(0.3)


    def forward(self, X):
        '''
        计算三层fcn
        '''
        #X = F.relu(self.fc1(X))
        #X = F.relu(self.fc2(X))
        #X=self.m(X)
        X1=F.softmax(X, dim=1)
        return X1,X