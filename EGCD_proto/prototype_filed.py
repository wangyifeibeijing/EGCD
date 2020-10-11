import networkx as nx
import torch
import itertools
import GCN_layers
import CDN_layers
import Loss_func
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G).todense()
#A需要正规化
A_normed = GCN_layers.normalize(torch.FloatTensor(A.astype("float32")),True)
#A_normed = normalize(torch.FloatTensor(A),True)

N = len(A)
X_dim = N

# 没有节点的特征，简单用一个单位矩阵表示所有节点
X = torch.eye(N,X_dim)
# 正确结果
Y = torch.zeros(N,1).long()
# 计算loss的时候要去掉没有标记的样本
Y_mask = torch.zeros(N,1,dtype=torch.uint8)
# 一个分类给一个样本
Y[0][0]=0
Y[N-1][0]=1
#有样本的地方设置为1
Y_mask[0][0]=1
Y_mask[N-1][0]=1

#真实的空手道俱乐部的分类数据
Real = torch.zeros(34 , dtype=torch.long)
for i in [1,2,3,4,5,6,7,8,11,12,13,14,17,18,20,22] :
	Real[i-1] = 0
for i in [9,10,15,16,19,21,23,24,25,26,27,28,29,30,31,32,33,34] :
	Real[i-1] = 1

# 我们的GCN模型
gcn = GCN_layers.GCN(A_normed ,X_dim,2)
cdn= CDN_layers.CDN(16,2)
#选择adam优化器
gd = torch.optim.Adam(itertools.chain(gcn.parameters(),cdn.parameters()))

for i in range(300):
	#转换到概率空间
	y_pred =GCN_layers.F.softmax(cdn(gcn(X)),dim=1)
	k, mi = y_pred.max(1)
	one_hot = torch.eye(2)[mi, :]

	criterion = Loss_func.ETR_loss()

	loss = criterion(A_normed, one_hot)
	loss.requires_grad_(True)


	#梯度下降
	#清空前面的导数缓存
	gd.zero_grad()
	#求导
	loss.backward()
	#一步更新
	gd.step()

	if i%20==0 :

		print(mi)

		print(one_hot)
		#计算精确度
		print((mi == Real).float().mean())