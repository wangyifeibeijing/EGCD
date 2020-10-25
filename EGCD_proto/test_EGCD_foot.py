import EGCD_prototype
import data_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics





features,Amatrix,labels=data_loader.load_football()
features=np.array(features.astype("float32"))
Amatrix=np.array(Amatrix.astype("float32"))
labels=np.array(labels.astype("float32"))
'''
print(features.shape,Amatrix.shape,labels)
'''
n_select=500
features =features[:n_select,:]
Amatrix=Amatrix[:n_select,:n_select]
labels=labels[:n_select,:]
labels_true = np.argmax(labels, axis=1)

mi,loss_ram,y_original,H=EGCD_prototype.use_EGCD(features,Amatrix,labels)

plt.plot(loss_ram)
plt.title('Loss value')
plt.xlabel('times')
plt.ylabel('loss value')
plt.show()
A=np.array(mi)

print(A)
print(labels_true)
'''
print(labels_true.shape)
'''
print('proposed')
print(metrics.normalized_mutual_info_score(A,labels_true))



from sklearn.cluster import KMeans
km = KMeans(n_clusters=12)   #将数据集分为2类
y_pre = km.fit_predict(features)
print('kmeans')
print(metrics.normalized_mutual_info_score(y_pre,labels_true))
from sklearn.cluster import SpectralClustering
y_pre = SpectralClustering(n_clusters=12).fit_predict(features)
print('ncut')
print(metrics.normalized_mutual_info_score(y_pre,labels_true))

h=labels_true.__len__()
from sklearn.cluster import KMeans
km = KMeans(n_clusters=12)   #将数据集分为2类
y_pre = km.fit_predict(features)
arr = np.random.rand(h,1)
label = y_pre
for i in range(h):
    k= int(arr[i]*120)/10
    label[i]=k
label=label
labels_true=np.array(labels_true)
print('rand')
print(metrics.normalized_mutual_info_score(label,labels_true))