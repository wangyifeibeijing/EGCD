import torch
import numpy as np
import itertools
from EGCD_v_0_3.EGCD_parts import GCN_layers
from EGCD_v_0_3.EGCD_parts import Loss_func
from EGCD_v_0_3.EGCD_parts import CDN_layers
import torch.nn as nn
def k_neighboors(XXT,k):
	XN = XXT.copy()
	XN.sort(0)

	[n, _] = XXT.shape
	k=np.min([n-3,k])
	for i in range(n):
		for j in range(n):
			if (XXT[i, j] < XN[-(k+1), j]):
				XXT[i, j] = 0
	return XXT
def L2_distance_1(X,Y):
	[n,m]=X.shape
	rep=np.ones((1,n))
	XX = np.mat(np.sum(np.multiply(X ,X ),axis=1)).T.dot(rep)
	YY = np.mat(np.sum(np.multiply(Y, Y), axis=1)).T.dot(rep)
	XY=X.dot(Y.T)
	W=XX+YY-2*XY
	return W
def W_RBF(W,neigh):
	W=k_neighboors(W, neigh)
	W = 0.5 * (W + W.T)
	sigma=W.max()
	RBF=np.exp((-1*W)/sigma)
	return RBF
def constru_AW(A,X,neigh,feat_map='dot',mask=False):

	flag=0#标注是否为带权拓扑结构
	if(feat_map=='dot'):
		flag=1
		XXT=X.dot(X.T)
		XXT = k_neighboors(XXT, neigh)
		XXT=0.5*(XXT+XXT.T)


	elif (feat_map == 'rbf'):
		flag=1
		W=L2_distance_1(X,X)
		XXT=W_RBF(W,neigh)


	else:
		XXT=A
	if(mask):
		XXT=np.multiply(A, XXT)
	[n, _] = XXT.shape
	for i in range(n):
		XXT[i,i]=0.0
	if(flag==10):
		Amatrix = np.mat(XXT)
		[n,_]=Amatrix.shape
		for i in range(n):
			Amatrix[i,i]=0.0
		degree = np.sum(Amatrix, axis=0)
		# print(degree.tolist())
		degree = degree.tolist()[0]
		degree =(np.power(degree,-0.5))
		D = np.diag(degree)
		XXT = D.dot(Amatrix.dot(D))


	return XXT

def use_EGCD(features_in,adj_in,labels_in,feat_map='dot',neighbors=20,mask=False,printloss=False):

	A = adj_in.astype("float32")
	A_0=constru_AW(A,features_in,neighbors,feat_map,mask)

	#A需要正规化
	A0=torch.FloatTensor(A_0.astype("float32"))
	A1=GCN_layers.normalize(torch.FloatTensor(A_0.astype("float32")),True)#邻接矩阵转tensor
	A_normed = GCN_layers.normalize(torch.FloatTensor(A.astype("float32")),True)#对A正规化
	X=torch.Tensor(features_in)#features 转 tensor
	_,X_dim = X.shape

	[n,k]=labels_in.shape#获得labels的维度来获取类数
	# 我们的GCN模型
	gcn = GCN_layers.GCN(A_normed ,X_dim,k)#GCN部分
	cdn= CDN_layers.CDN(k,k)#全连接部分
	#选择adam优化器
	gd = torch.optim.Adam(itertools.chain(gcn.parameters(),cdn.parameters()),lr=0.1,amsgrad=True)#,weight_decay=0.1,amsgrad=True	loss_ram=[]#训练loss存储
	#训练
	loss_ram=[]

	for i in range(1000):
		#转换到概率空间
		H=gcn(X)
		y_pred ,y_original =cdn(H)
		k, y_out = y_pred.max(1)
		criterion = Loss_func.ETR_loss_trace()
		loss = criterion(A0, y_pred)
		'''
		if(i>=150):
			if i%200==0:
				for p in gd.param_groups:
					p['lr'] *=0.1
		'''
		if (i >= 20):
			for p in gd.param_groups:
				p['lr'] = 0.01
		if (i >= 100):
			for p in gd.param_groups:
				p['lr'] = 0.001
		if i>=200:
			if i % 50 == 0:
				for p in gd.param_groups:
					p['lr'] = 1/(1+0.1*i)*0.001
		loss.requires_grad_(True)
		#梯度下降
		#清空前面的导数缓存
		gd.zero_grad()
		#求导
		loss.backward()
		#单步更新
		gd.step()

		loss_ram.append(loss)
		if(printloss):
			print(loss)


	return y_out,loss_ram,y_original,H#返回:输出（单一社区），loss记录，输出前层（重叠社区），GCN输出

