import torch
import torch.nn as nn
import torch.nn.functional as F

class CDN(nn.Module):
    def __init__(self,  dim_in, dim_out):
        super(CDN, self).__init__()
        #self.H = H
        self.fc1 = nn.Linear(dim_in, dim_in, bias=False)
        self.fc2 = nn.Linear(dim_in, dim_out, bias=False)
        self.m = nn.Dropout(0.5)

    def forward(self, X):
        '''
        计算三层gcn
        '''
        #X1 = F.relu(self.fc1(X))
        X1 = F.relu(self.fc2(X))
        X2=self.m(X1)
        return X2