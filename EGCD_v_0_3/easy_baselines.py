from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import numpy as np
from sklearn import metrics
from EGCD_v_0_3 import data_loader
from EGCD_v_0_3 import Acc_calculator
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
    return y_pre

def rand_deal(dataname,features,Amatrix,labels):
    h,k=labels.shape
    #labels_true = np.argmax(labels, axis=1)

    label = np.random.randint(0,k,size=h).tolist()
    return label

def te_rand(dataname):
    #加载DATASET
    features,Amatrix,labels=data_loader.load_which(dataname)
    features=np.array(features.astype("int"))
    Amatrix=np.array(Amatrix.astype("int"))
    labels=np.array(labels.astype("int"))
    [n, k] = labels.shape
    labels_true = np.argmax(labels, axis=1)#从one-hot计算真实label

    nmi_list=[]
    acc_list=[]
    [size,d]=features.shape
    nmi_times=15
    for i in range(nmi_times):
            #print(i)
            label=rand_deal(dataname,features,Amatrix,labels)

            A=np.array(label)#将输出label转nparray
            nmi=metrics.normalized_mutual_info_score(A, labels_true)
            acc = Acc_calculator.use_acc(labels_true, A)
            # print(A)

            # print(nmi)
            nmi_list.append(nmi)
            acc_list.append(acc)
    meani = np.mean(nmi_list)
    vari = np.var(nmi_list)
    # print(str(meani)+'+'+str(vari))
    mean_nmi = round(float(meani), 4)
    mean_acc = round(float(np.mean(acc_list)), 4)
    print(str(mean_acc) + '\t' + dataname + '\t' + str(mean_nmi))


def te_kmeans(dataname):
    #加载DATASET
    features,Amatrix,labels=data_loader.load_which(dataname)
    features=np.array(features.astype("int"))
    Amatrix=np.array(Amatrix.astype("int"))
    labels=np.array(labels.astype("int"))
    [n, k] = labels.shape
    labels_true = np.argmax(labels, axis=1)#从one-hot计算真实label

    nmi_list=[]
    acc_list=[]
    [size,d]=features.shape
    if(size<1000):
        nmi_times=5
    elif(size<2000):
        nmi_times = 3
    else:
        nmi_times = 1
    for i in range(nmi_times):
            #print(i)
            label=kmeans_deal(dataname,features,Amatrix,labels)

            A=np.array(label)#将输出label转nparray
            nmi=metrics.normalized_mutual_info_score(A, labels_true)
            acc = Acc_calculator.use_acc(labels_true, A)
            # print(A)

            # print(nmi)
            nmi_list.append(nmi)
            acc_list.append(acc)
    meani = np.mean(nmi_list)
    vari = np.var(nmi_list)
    # print(str(meani)+'+'+str(vari))
    mean_nmi = round(float(meani), 4)
    mean_acc = round(float(np.mean(acc_list)), 4)
    print(str(mean_acc) + '\t' + dataname + '\t' + str(mean_nmi))

if __name__ == '__main__':

    name_list = [
        'cornell',
        'texas',
        'washington',
        'wisconsin',

        'cora',
        'TerrorAttack',
        'citeseer',
        'Pubmed_small',

    ]
    '''
    for dataname in name_list:
        # print('using infomap on ' + dataname)
        te_kmeans(dataname)
    print('=======================================================')
    '''
    for dataname in name_list:
        # print('using infomap on ' + dataname)
        te_rand(dataname)