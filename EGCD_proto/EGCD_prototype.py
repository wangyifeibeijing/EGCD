import networkx as nx
import torch
import itertools
import GCN_layers
import CDN_layers
import Loss_func
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import datetime
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
from torch.nn import init
#define the initial function to init the layer's parameters for the network

def z_score(x, axis=0):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x
def use_EGCD(features_in,adj_in,labels_in):

	A = adj_in
	#A需要正规化
	A1=torch.FloatTensor(A.astype("float32"))
	A_normed = GCN_layers.normalize(torch.FloatTensor(A.astype("float32")),True)
	N = len(A)
	#features_in=z_score(features_in)
	X=torch.Tensor(features_in)
	_,X_dim = X.shape
	#获得labels的维度来获取类数
	[n,k]=labels_in.shape
	# 我们的GCN模型
	gcn = GCN_layers.GCN(A_normed ,X_dim,k)

	cdn= CDN_layers.CDN(k,k)
	#选择adam优化器
	gd = torch.optim.Adam(itertools.chain(gcn.parameters()),lr=0.01)#,weight_decay=0.1,amsgrad=True
	#CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(gd, T_max=10, eta_min=0)
	loss_ram=[]
	for i in range(1000):
		#转换到概率空间
		H=gcn(X)
		y_pred ,y_original =cdn(H)
		k, mi = y_pred.max(1)

		criterion = Loss_func.ETR_loss_trace()
		loss = criterion(A1, y_pred)
		loss.requires_grad_(True)
		#梯度下降
		#清空前面的导数缓存
		gd.zero_grad()
		#求导
		loss.backward()
		#一步更新
		gd.step()
		#CosineLR.step()
		loss_ram.append(loss)
		print(H)
		'''
		if i%10==0 :
			if i>=20:
				for p in gd.param_groups:
					p['lr'] *= 0.9
		
			#lr_list.append(gd.state_dict()['param_groups'][0]['lr'])
			#print((mi == Real).float().mean())
			'''

	return mi,loss_ram,y_original,H