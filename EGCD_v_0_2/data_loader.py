# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.io as sio
# 导入数据：分隔符为空格
from igraph  import *

def load_which(name):
	if (name=='cora'):
		features, Amatrix, labels=load_cora()#load dataset cora
	elif (name=='dblp'):
		features, Amatrix, labels=load_dblp()#dblp unfinished
	elif (name == 'football'):
		features, Amatrix, labels = load_football()#load dataset football
	elif (name == 'polbooks'):
		features, Amatrix, labels = load_polbooks()  # load dataset football
	elif (name == 'six'):
		features, Amatrix, labels = load_six()#load dataset with six points, no overlapping part
	elif (name == 'five_overlap'):
		features, Amatrix, labels = load_five_overlap()#load dataset with five points, one overlapping part
	elif (name == 'polblogs'):
		features, Amatrix, labels = load_polblogs()#load dataset polblogs
	elif (name == 'citeseer'):
		features, Amatrix, labels = load_citeseer()  # load dataset polblogs
	elif (name == 'cornell'):
		features, Amatrix, labels = load_cornell()  # load dataset cornell
	elif (name == 'texas'):
		features, Amatrix, labels = load_texas()  # load dataset texas
	elif (name == 'washington'):
		features, Amatrix, labels = load_washington()  # load dataset washington
	elif (name == 'wisconsin'):
		features, Amatrix, labels = load_wisconsin()  # load dataset wisconsin
	elif (name == 'TerrorAttack'):
		features, Amatrix, labels = load_TerrorAttack()  # load dataset TerrorAttack
	elif (name == 'email_EU'):
		features, Amatrix, labels = load_email_EU()  # load dataset email_EU
	elif (name == 'Pubmed_small'):
		features, Amatrix, labels = load_Pubmed_small()  # load dataset email_EU
	else:
		print('no dataset named '+name+', auto exit')
		features = []
		Amatrix = []
		labels = []
		sys.exit(0)  # os._exit() 用于在线程中退出,sys.exit()用于在主线程中退出，exit(0)#终止退出程序，会关闭窗口
	return features, Amatrix, labels
def load_cora():
	raw_data = pd.read_csv('../data/cora/cora.content',sep = '\t',header = None)
	num = raw_data.shape[0] # 样本点数2708
	# 将论文的编号转[0,2707]
	a = list(raw_data.index)
	b = list(raw_data[0])
	c = zip(b,a)
	map = dict(c)
	# 将词向量提取为特征,第二行到倒数第二行
	features =raw_data.iloc[:,1:-1]
	 # 检查特征：共1433个特征，2708个样本点
	#print(features.shape)
	labels = pd.get_dummies(raw_data[1434])
	#print(labels.shape)

	#导入论文引用数据
	raw_data_cites = pd.read_csv('../data/cora/cora.cites',sep = '\t',header = None)
	# 创建一个规模和邻接矩阵一样大小的矩阵
	Amatrix = np.zeros((num,num))
	# 创建邻接矩阵
	for i ,j in zip(raw_data_cites[0],raw_data_cites[1]):
		x = map[i] ; y = map[j]  #替换论文编号为[0,2707]
		Amatrix[x][y] = Amatrix[y][x] = 1 #有引用关系的样本点之间取1
	# 查看邻接矩阵的元素和（按每列汇总）
	#print((Amatrix.shape))

	return features,Amatrix,labels
def load_dblp():
	features=[]
	Amatrix=[]
	labels=[]
	return features, Amatrix, labels
def load_football():
	path='../data/football/football.gml'
	g = Graph.Read_GML(path)
	Amatrix = g.get_adjacency()

	Amatrix = np.array(Amatrix.data)

	labels_temp=(g.vs['value'])

	[n, d] = Amatrix.shape
	features = np.eye(n)
	labels=np.zeros((n,12))
	for i in range(n):
		for j in range(12):
			if(labels_temp[i]==j):
				labels[i,j]=1.0
	#print(features.shape)
	return features, Amatrix, labels
def load_polbooks():
	path='../data/polbooks/polbooks.gml'
	g = Graph.Read_GML(path)
	Amatrix = g.get_adjacency()

	Amatrix = np.array(Amatrix.data)

	labels_temp=(g.vs['value'])

	[n, d] = Amatrix.shape
	features = np.eye(n)
	labels=np.zeros((n,3))
	for i in range(n):
		if(labels_temp[i]=='l'):
			labels[i,0]=1.0
		elif (labels_temp[i] == 'n'):
			labels[i, 1] = 1.0
		else:
			labels[i, 2] = 1.0
	return features, Amatrix, labels
def load_six():
	features=np.eye(6)
	Amatrix=[
		[0, 1, 1, 0, 0, 0],
		[1, 0, 1, 0, 0, 0],
		[1, 1, 0, 1, 0, 0],
		[0, 0, 1, 0, 1, 1],
		[0, 0, 0, 1, 0, 1],
		[0, 0, 0, 1, 1, 0],
			 ]
	Amatrix=np.mat(Amatrix)
	labels=[[0,0,0,1,1,1],
			[1,1,1,0,0,0]]
	labels=np.array(labels).T
	return features, Amatrix, labels
def load_five_overlap():
	features=np.eye(5)
	Amatrix=[
		[0, 1, 1, 0, 0],
		[1, 0, 1, 0, 0],
		[1, 1, 0, 1, 1],
		[0, 0, 1, 0, 1],
		[0, 0, 1, 1, 0],
			 ]
	Amatrix=np.mat(Amatrix)
	labels=[[0,0,0.5,1,1],
			[1,1,0.5,0,0]]
	labels=np.array(labels).T
	return features, Amatrix, labels
def load_polblogs():

	path = '../data/polblogs/polblogs.gml'
	g = Graph.Read_GML(path)
	Amatrix = g.get_adjacency()

	Amatrix = np.array(Amatrix.data)


	labels_temp = (g.vs['value'])

	[n, d] = Amatrix.shape
	features=np.eye(n)
	labels = np.zeros((n, 2))
	for i in range(n):
		for j in range(2):
			if (labels_temp[i] == j):
				labels[i, j] = 1.0
	#print(features.shape)
	return features, Amatrix, labels
def load_citeseer():
	cs_content = pd.read_csv('../data/citeseer/citeseer.content', sep='\t', header=None, low_memory=False)
	#print(cs_content.shape)
	cs_cite = pd.read_csv('../data/citeseer/citeseer.cites', sep='\t', header=None, low_memory=False)
	#print(cs_cite.shape)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]  # 论文id全部转换为string,paper_id不都是整数值，惊了！
	mp = dict(zip(paper_id, ct_idx))
	#print(mp['zamir99grouper'])
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))

	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():  # 数据集有问题！！在cites中有未出现过的paper_id
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def load_cornell():
	cs_content = pd.read_csv('../data/WebKB/cornell.content', sep='\t', header=None)
	cs_cite = pd.read_csv('../data/WebKB/cornell.cites', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def load_texas():
	cs_content = pd.read_csv('../data/WebKB/texas.content', sep='\t', header=None)
	cs_cite = pd.read_csv('../data/WebKB/texas.cites', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def load_washington():
	cs_content = pd.read_csv('../data/WebKB/washington.content', sep='\t', header=None)
	cs_cite = pd.read_csv('../data/WebKB/washington.cites', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def load_wisconsin():
	cs_content = pd.read_csv('../data/WebKB/wisconsin.content', sep='\t', header=None)
	cs_cite = pd.read_csv('../data/WebKB/wisconsin.cites', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label

def load_TerrorAttack():
	cs_content = pd.read_csv('../data/TerrorAttack/terrorist_attack.nodes', sep='\t', header=None)
	cs_cite = pd.read_csv('../data/TerrorAttack/terrorist_attack_loc_org.edges', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = cs_content.iloc[:, 1:-1]
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def load_email_EU():
	cs_content = pd.read_csv('../data/email_EU/email-Eu-core-department-labels.txt', sep=';', header=None)
	cs_cite = pd.read_csv('../data/email_EU/email-Eu-core.txt', sep=';', header=None)
	ct_idx = list(cs_content.index)
	paper_id = list(cs_content.iloc[:, 0])
	paper_id = [str(i) for i in paper_id]
	mp = dict(zip(paper_id, ct_idx))
	label = cs_content.iloc[:, -1]
	label = pd.get_dummies(label)
	#print(label.shape)
	feature = np.eye(1005)
	#print(feature.shape)
	mlen = cs_content.shape[0]
	adj = np.zeros((mlen, mlen))
	for i, j in zip(cs_cite[0], cs_cite[1]):
		if str(i) in mp.keys() and str(j) in mp.keys():
			x = mp[str(i)]
			y = mp[str(j)]
			adj[x][y] = adj[y][x] = 1
	return feature, adj, label
def get_pubmed_list():
	path = '../data/Pubmed/Pubmed_feature_list.txt'
	file = open(path)
	c = file.readline()[2:-2]
	file.close()
	#print(c)
	list2 = c.split('\', \'')
	#print(list2)
	return list2
def load_Pumbed():
	feature_list=get_pubmed_list()
	d=len(feature_list)
	n=19717
	X=np.zeros((n,d))
	key=np.zeros((n,1),'int')
	Y=np.zeros((n,3))
	path='../data/Pubmed/Pubmed-Diabetes.NODE.paper.tab'
	file = open(path)

	i=0
	while 1:

		line = file.readline()
		if not line:
			break
		if(line.find('label=')!=-1):
			temp = line.split('\t')
			for fea in temp:
				if(fea.find('summary')==-1) :
					if (fea.find('=')==-1):
						key[i,0]=int(fea)
					elif(fea.find('label')!=-1):
						part=fea.split('=')
						part1=part[0]
						part2=part[1]
						Y[i,int(part2)-1]=1
					else:
						part = fea.split('=')
						part1 = part[0]
						part2 = part[1]
						X[i,feature_list.index(part1)]=float(part2)
			i = i + 1
	file.close()
	key=key.T[0].tolist()
	A=read_pubmed_g(key)
	return X,A,Y

def read_pubmed_g(key):
	A=np.zeros((19717,19717))
	path = '../data/Pubmed/Pubmed-Diabetes.DIRECTED.cites.tab'
	file = open(path)
	flag=0
	while 1:

		line = file.readline()
		flag=flag+1
		if not line:
			break
		if(flag>2):
			temp = line.split('\t')
			startpaper=temp[1]
			st=startpaper.split(':')
			startid=int(st[1])
			endpaper=temp[3]
			ed = endpaper.split(':')
			endtid = int(ed[1])
			i=key.index(startid)
			j=key.index(endtid)
			A[i,j]=1
	file.close()
	return A
def load_Pubmed_small():
	path='../data/Pubmed/Pubmed_3934.mat'
	data=sio.loadmat(path)
	feature=data['feature']
	adj = data['adj']
	adj=adj+adj.T
	adj[adj>0]=1
	label = data['label']
	return feature, adj, label
def L2_distance_1(X,Y):
	[n,m]=X.shape
	rep=np.ones((1,n))
	XX = np.mat(np.sum(np.multiply(X ,X ),axis=1)).T.dot(rep)
	YY = np.mat(np.sum(np.multiply(Y, Y), axis=1)).T.dot(rep)
	XY=X.dot(Y.T)
	W=XX+YY-2*XY
	return W
def W_RBF(W):
	sigma=0.5*np.max(W)
	RBF=np.exp((-1*(0.5*(W+W.T))/sigma))
	return RBF
def constru_AW(A,X,feat_map='dot',mask=False):
	flag=0#标注是否为带权拓扑结构
	if(feat_map=='dot'):
		flag=1
		XXT=X.dot(X.T)

	elif (feat_map == 'rbf'):
		flag=1
		W=L2_distance_1(X,X)
		XXT=W_RBF(W)

	else:
		XXT=A
	if(mask):
		XXT=np.multiply(A, XXT)
	if(flag==1):
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

if __name__ == '__main__':


	'''
	name_list=[
		'six',
		#'dblp',
		'football',
		'polbooks',
		#'five_overlap',
		'cornell',
		'texas',
		'washington',
		'wisconsin',
		'email_EU',
		'TerrorAttack',
		'polblogs',
		'cora',
		'citeseer',
		'Pubmed_small',


		'fake_name_test'
			   ]
	nofe_name_list = [
		'six',
		# 'dblp',
		'football',
		'polbooks',
		# 'five_overlap',

		'email_EU',

		'polblogs',

	]

	for name in name_list:
		print(name+':')
		x,a,y=load_which(name)

		print(x.shape,y.shape,0.5*np.sum(a))
	'''
	XXT = np.random.rand(10,10)
	print(XXT)

	print(XXT)