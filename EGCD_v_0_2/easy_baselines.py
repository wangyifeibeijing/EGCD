from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn import metrics
import data_loader

def kmeans_deal(dataname,features,Amatrix,labels):
    [n,m]=features.shape
    if n==m:
        print('kmeans on ' + dataname + ':')
        print('no real feature')
        return -1
    [n, k] = labels.shape
    labels_true = np.argmax(labels, axis=1)
    km = KMeans(n_clusters=k)   #将数据集分为k类
    y_pre = km.fit_predict(features)
    print('kmeans on '+dataname+':')
    print(metrics.normalized_mutual_info_score(y_pre,labels_true))
def ncut_deal(dataname,features,Amatrix,labels):
    [n, m] = Amatrix.shape
    if n >1500:
        print('ncut on ' + dataname + ':')
        print('too large')
        return -1
    lambda1, features = np.linalg.eig(Amatrix)
    [n, k] = labels.shape
    labels_true = np.argmax(labels, axis=1)
    km = KMeans(n_clusters=k)  # 将数据集分为k类
    y_pre = km.fit_predict(features)
    print('ncut on '+dataname+':')
    print(metrics.normalized_mutual_info_score(y_pre, labels_true))
def rand_deal(dataname,features,Amatrix,labels):
    labels_true = np.argmax(labels, axis=1)
    h = labels_true.__len__()
    label = labels_true.copy()
    arr = np.random.rand(h, 1)
    for i in range(h):
        k = int(arr[i] * 20) / 10
        label[i] = k
    print('rand on '+dataname+':')
    print(metrics.normalized_mutual_info_score(label, labels_true))



def use_all(dataname,features, Amatrix, labels):
    features = np.array(features.astype("float32"))
    Amatrix = np.array(Amatrix.astype("float32"))
    labels = np.array(labels.astype("float32"))
    labels_true = np.argmax(labels, axis=1)
    #rand_deal(dataname,features, Amatrix, labels)
    kmeans_deal(dataname, features, Amatrix, labels)
    #ncut_deal(dataname, features, Amatrix, labels)

if __name__ == '__main__':


    dataname = 'Pubmed_small'
    features, Amatrix, labels = data_loader.load_wisconsin()
    use_all(dataname, features, Amatrix, labels)