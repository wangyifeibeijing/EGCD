import data_loader
import numpy as np
from sklearn import metrics
import Leading_vec as LE
def te_LE(dataname):
    #加载DATASET
    features,Amatrix,labels=data_loader.load_which(dataname)
    features=np.array(features.astype("float32"))
    Amatrix=np.array(Amatrix.astype("float32"))
    labels=np.array(labels.astype("float32"))
    [n, k] = labels.shape
    labels_true = np.argmax(labels, axis=1)#从one-hot计算真实label
    #print(labels_true)

    nmi_list=[]

    [size,d]=features.shape
    if(size<1000):
        nmi_times=5
    elif(size<2000):
        nmi_times = 3
    else:
        nmi_times = 1
    for i in range(nmi_times):
            #print(i)
            label=LE.use_Leading_vec(Amatrix,k, weighted=False)

            A=np.array(label)#将输出label转nparray
            nmi=metrics.normalized_mutual_info_score(A, labels_true)
            #print(A)

            #print(nmi)
            nmi_list.append(nmi)
    meani = np.mean(nmi_list)
    vari = np.var(nmi_list)
    print(str(meani)+'+'+str(vari))
if __name__ == '__main__':

    name_list = [
        'Pubmed_small',


    ]


    for dataname in name_list:
        print('using LE on ' + dataname)
        te_LE(dataname)
