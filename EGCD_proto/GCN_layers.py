import torch
import torch.nn as nn
import torch.nn.functional as F




def normalize(A , symmetric=True):
	# A = A+I
	A = A + torch.eye(A.size(0))
	# 所有节点的度
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)


class GCN(nn.Module):
	'''
	Z = AXW
	'''
	def __init__(self , A, dim_in , dim_out):
		super(GCN,self).__init__()

		self.A = A
		self.fc1 = nn.Linear(dim_in ,32,bias=False)
		#self.fc1_1 = nn.Linear(256, 128, bias=False)
		#self.fc1_2 = nn.Linear(128, 64, bias=False)
		self.fc2 = nn.Linear(32,dim_out,bias=False)
		#self.fc3 = nn.Linear(32,16,bias=False)

	def forward(self,X):
		'''
		计算三层gcn
		'''

		X = F.leaky_relu(self.fc1(self.A.mm(X)))
		#X = F.relu(self.fc1_1(self.A.mm(X)))
		#X = F.relu(self.fc1_2(self.A.mm(X)))
		X = F.leaky_relu(self.fc2(self.A.mm(X)))
		return X

#获得空手道俱乐部数据
