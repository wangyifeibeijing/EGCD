from data_set import data_loader
import numpy as np
from sklearn import metrics
from baselines import Infomap as infomap
from basic_files import Acc_calculator
def te_infomap(dataname):
    #加载DATASET
    features,Amatrix,labels=data_loader.load_fast(dataname)
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
            label=infomap.use_infomap(Amatrix,k, weighted=False)

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
    mean_acc = round(float(np.max(acc_list)), 4)
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


    for dataname in name_list:
        #print('using infomap on ' + dataname)
        te_infomap(dataname)
