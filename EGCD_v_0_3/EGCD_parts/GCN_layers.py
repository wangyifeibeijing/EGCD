import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(A , symmetric=True):
	A = A + torch.eye(A.size(0))# A = A+I
	d = A.sum(1)# 所有节点的度
	if symmetric:
		D = torch.diag(torch.pow(d , -0.5))#D = D^-1/2
		return D.mm(A).mm(D)
	else :
		D =torch.diag(torch.pow(d,-1))# D=D^-1
		return D.mm(A)


class GCN(nn.Module):
	'''
	Z = AXW
	'''
	def __init__(self , A, dim_in , dim_out):
		super(GCN,self).__init__()
		self.A = A
		self.norm1 = nn.BatchNorm1d(dim_in)
		self.fc1 = nn.Linear(dim_in ,64,bias=False)
		self.norm2 = nn.BatchNorm1d(64)
		self.fc2 = nn.Linear(64,dim_out,bias=False)



	def forward(self,X):
		'''
		计算gcn
		'''
		X=self.norm1(X)
		X = F.leaky_relu(self.fc1(self.A.mm(X)))
		#X = F.relu(self.fc1(self.A.mm(X)))
		X = self.norm2(X)
		X = F.leaky_relu(self.fc2(self.A.mm(X)))
		#X = F.relu(self.fc2(self.A.mm(X)))
		return X


