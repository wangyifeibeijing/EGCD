import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import torch
import scipy.sparse as sp
import numpy as np
import torch.optim as optim
import gc

import matplotlib.pyplot as plt
from sklearn import metrics
from data_set import data_loader
class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=False):
        '''

        :param input_dim: int
        :param output_dim: int
        :param use_bias: bool, optional
        '''
        super(GraphConvolution, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.ues_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.ues_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias',None)
        #self.reset_parameters()
    def reset_parameters(self):
        init.xavier_uniform_(self.weight,gain = 1.0)
        #init.kaiming_uniform_(self.weight,a=0, mode='fan_in', nonlinearity='leaky_relu')
        if self.ues_bias:
            init.zeros_(self.bias)
    def forward(self, adjacency,input_feature):
        '''

        :param adjacency: n*n tensor#.SPARSE
        :param input_feature: n*input_dim
        :return: output:n*output_dim
        '''
        support=torch.mm(input_feature,self.weight)
        output=torch.mm(adjacency,support)
        #output = torch.sparse.mm(adjacency, support)
        if self.ues_bias:
            output+=self.bias
        return output


class EGCD_Net(nn.Module):
    '''
    EGCD:
    '''
    def __init__(self,input_dim,out_put_dim):
        super(EGCD_Net,self).__init__()
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.gcn1=GraphConvolution(input_dim,64)
        self.norm2 = nn.BatchNorm1d(64)
        self.gcn2 = GraphConvolution(64, out_put_dim)
        self.norm3 = nn.BatchNorm1d(out_put_dim)
        self.gcn3 = GraphConvolution(out_put_dim, out_put_dim)
        self.norm4 = nn.BatchNorm1d(out_put_dim)
    def forward(self, adjacency,feature):
        feature=self.norm1(feature)
        h1=F.leaky_relu(self.gcn1(adjacency,feature))#leaky_relu
        h1=self.norm2(h1)
        h2=F.leaky_relu(self.gcn2(adjacency,h1))#leaky_relu
        h2 = self.norm3(h2)

        '''
       
        h2 = self.gcn3(adjacency, h2)
        h2 = self.norm4(h2)
        '''
        result = F.softmax(h2, dim=1)
        return result


class EGCD_Net_L(nn.Module):
    '''
    EGCD:
    '''

    def __init__(self, input_dim, out_put_dim):
        super(EGCD_Net_L, self).__init__()
        self.norm1 = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim ,64,bias=False)
        self.norm2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64 ,out_put_dim,bias=False)
        #self.norm3 = nn.BatchNorm1d(out_put_dim)
        #self.gcn3 = GraphConvolution(out_put_dim, out_put_dim)
        #self.norm4 = nn.BatchNorm1d(out_put_dim)

    def forward(self, adjacency, feature):
        feature = self.norm1(feature)
        h1 = F.leaky_relu(self.fc1(adjacency.mm(feature)))  # leaky_relu
        h1 = self.norm2(h1)
        h2 = F.leaky_relu(self.fc2(adjacency.mm(h1)))  # leaky_relu
        #h2 = self.norm3(h2)

        '''

        h2 = self.gcn3(adjacency, h2)
        h2 = self.norm4(h2)
        '''
        result = F.softmax(h2, dim=1)
        return result
#用于图网络使用的正则化
def normalization(A , symmetric=True):
    [n,_]=A.shape
    A = A + np.eye(n)# A = A+I
    d = A.sum(1)# 所有节点的度
    if symmetric:
        D = np.diag(np.power(d , -0.5))#D = D^-1/2
        return D.dot(A).dot(D)
    else :
        D =np.diag(torch.power(d,-1))# D=D^-1
        return D.dot(A)
#用于loss使用的A
def k_neighboors(XXT,k):
    if(k<0):
        return XXT
    XN = XXT.copy()
    XN.sort(0)
    [n, _] = XXT.shape
    k=np.min([n-3,k])
    for i in range(n):
        for j in range(n):
            if (XXT[i, j] < XN[-(k+1), j]):
                XXT[i, j] = 0
    return XXT
def k_neighboors_a(XXT,adj,k):
    with torch.no_grad():
        if(k<0):
            return XXT
        XN = XXT.copy()
        XN.sort(0)
        [n, _] = XXT.shape
        k=np.min([n-3,k])
        for i in range(n):
            for j in range(n):
                if (XXT[i, j] < XN[-(k+1), j]) and adj[i,j]==0:
                    XXT[i, j] = 0
    return XXT
def constru_AW(A,X,neigh,feat_map='dot'):
    with torch.no_grad():
        if(feat_map=='dot'):
            XXT=X.dot(X.T)
            XXT = k_neighboors(XXT, neigh)
            XXT=0.5*(XXT+XXT.T)
        elif (feat_map == 'dot_a'):
            XXT = X.dot(X.T)
            XXT = k_neighboors_a(XXT,A, neigh)
            XXT = 0.5 * (XXT + XXT.T)
        elif (feat_map=='dot_b'):
            XXT=X.dot(X.T)
            A_USE=A.copy()
            A_USE[A_USE != 0] = -1
            A_USE[A_USE != -1] = 1
            A_USE[A_USE != 1] = 0
            XXC=np.multiply(XXT, A_USE)
            A_S = k_neighboors(XXC, neigh)
            A_all=A_S+A
            A_all[A_all!=0]=1
            XXT = np.multiply(XXT, A_all)
        else:
            XXT=A

        [n, _] = XXT.shape
        for i in range(n):
            XXT[i,i]=0.0
        return XXT
def to_ones(features_in):
    features_in=np.mat(features_in)

    xx= np.multiply(features_in, features_in)
    #print(xx)
    xs=xx.sum(1)
    #print(xs)
    [_,d]=features_in.shape
    xtt=np.sqrt( np.tile(xs,(1,d)))
    #print(xtt)
    x=features_in/xtt
    return x
class ETR_loss_trace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, C):
        n, k = list(C.size())
        IsumC = torch.ones(1,n)
        IsumCDC = torch.ones(k, 1)
        total=torch.trace((1/(torch.sum(A)))*(C.t().mm(A.mm(C))).mul(torch.log2(IsumCDC.mm(IsumC.mm((A.mm(C)))*(1/(torch.sum(A))))+1e-40)))
        #constraint = 0.001*(torch.norm( (k/n)*(C.t().mm(C)) - torch.eye(k)))
        #total=total + constraint
        return total

def use_EGCD(features_in,adj_in,c_num,feat_map='dot',neigh=20):
    #基础设置，学习率、迭代次数
    learning_rate=0.1
    epochs=300
    #weight_decay=5e-4
    #自动寻找gpu
    #device="cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    _, X_dim = features_in.shape
    model=EGCD_Net_L(X_dim,c_num).to(device)
    #loss改为自定义loss
    criterion = ETR_loss_trace()
    #Adam优化器
    optimizer=optim.Adam(model.parameters(),lr=learning_rate,amsgrad=True)
    lr_optim=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=8, verbose=False,
                                               threshold=0.000001, threshold_mode='rel', cooldown=0, min_lr=1e-4, eps=1e-08)
    #数据处理
    x=features_in#/features_in.sum(1,keepdims=True)
    x = x.astype(np.float32)
    tensor_x=torch.from_numpy(x).to(device)
    normalize_adj=normalization(adj_in)
    normalize_adj = normalize_adj.astype(np.float32)
    tensor_adj=torch.from_numpy(normalize_adj).to(device)
    #x_dot=to_ones(features_in)
    loss_adj=constru_AW(normalize_adj, features_in, neigh, feat_map)
    loss_adj = loss_adj.astype(np.float32)
    tensor_loss_adj = torch.from_numpy(loss_adj).to(device)
    #记录组
    loss_history=[]
    model.train()
    for epoch in range(epochs):
        gc.collect()
        result=model(tensor_adj,tensor_x)
        loss = criterion(tensor_loss_adj, result)
        _, y_out = result.max(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_optim.step(loss)
        loss_history.append(loss.item())
        '''if (epoch >= 20):
            for p in optimizer.param_groups:
                p['lr'] = 0.01
        if (epoch >= 100):
            for p in optimizer.param_groups:
                p['lr'] = 0.001
        if epoch >= 200:
            if epoch % 50 == 0:
                for p in optimizer.param_groups:
                    p['lr'] = 1 / (1 + 0.1 * epoch) * 0.001'''
        '''
        if epoch % 10 == 0:
            for p in optimizer.param_groups:
                p['lr'] = 1/1+(0.1*epoch) * 0.01
        '''
    model.eval()
    with torch.no_grad():
        result = model(tensor_adj, tensor_x)
        _, y_out = result.max(1)
    return y_out,loss_history


if __name__ == '__main__':
    x,a,y=data_loader.load_fast('cora')
    print(x.shape)
    print(a.shape)
    print(y.shape)
    [_,k]=y.shape
    label,loss=use_EGCD(x, a, k,feat_map='dot',neigh=25)
    print(loss)
    print(label)
    labels_true = np.argmax(y, axis=1)
    nmi = metrics.normalized_mutual_info_score(label, labels_true)
    print((nmi))
    loss_final=loss[-1]
    plt_la = ' l:' + str(loss_final)
    plt.plot(loss, label=plt_la)
    plt.xlabel('times')
    plt.ylabel('loss value')
    plt.legend(loc='upper right')
    plt.title('Loss value ')
    #pltname = path + '/' + dataname + name_part + '.png'
    #plt.savefig(pltname)
    plt.show()
    plt.cla()