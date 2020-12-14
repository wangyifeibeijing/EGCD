import numpy as np
import random
from data_set import data_loader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
import scipy.io as scio
import matplotlib.pyplot as plt

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    #y_true = y_true.astype(np.int64)

    assert len(y_pred) == len(y_true)
    D = max(np.max(y_pred), np.max(y_true)) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind,tid = linear_assignment(np.max(w) - w)

    flag=0
    for i in range(len(y_pred)):
        if(ind[y_true[i]]==tid[y_pred[i]]):
            flag=flag+1
    return flag/len(y_pred)
def list2onehot(label,k):
    _,n=label.shape
    onehot=np.zeros((n,k))
    for i in range(n):
        for j in  range(k):
            if(label[0,i]==j):
                onehot[i,j]=1
    #print(onehot)
    return onehot

def use_acc(label_true, label_proposed):#true first
    la_aim=bestmap(label_proposed,label_true)
    acc=accuracy_score(label_true, la_aim)
    return acc

def bestmap(label_pro,lable_true):
    label_out=[]
    cost_matrix=construct_W(label_pro,lable_true)
    matches,changes = linear_assignment(cost_matrix)
    for i in label_pro:
        label_out.append(changes[i])
    return label_out

def construct_W(label_pro,lable_true):
    label_pro=np.mat(label_pro)
    lable_true = np.mat(lable_true)
    label_pro = label_pro - label_pro.min()
    lable_true = lable_true - lable_true.min()
    first_clu = lable_true.min()
    last_clu = lable_true.max()
    first_clu1 = label_pro.min()
    last_clu1 = label_pro.max()
    k = np.max([last_clu - first_clu + 1,last_clu1 - first_clu1 + 1])
    label_pro_one=list2onehot(label_pro, k)
    lable_true_one = list2onehot(lable_true, k)
    #W=np.zeros((k,k))
    W=label_pro_one.T.dot(lable_true_one)
    W=W.max()+10-1*W
    return W

def test_list_gene(k_t,leni,clu):
    list_gron=[]
    list_use=[]
    list_true = np.random.randint(0,clu,size=leni).tolist()
    for i in range(leni):
        list_gron.append(list_true[i])
    changed = random.sample(range(0, leni), k_t)
    for i in changed:
        k=random.sample(range(0, clu), 1)[0]
        if k!=list_gron[i]:
            list_gron[i]=k
        elif k>0:
            list_gron[i] = k-1
        elif k<leni:
            list_gron[i] = k + 1
    change_list=index = random.sample(range(0,clu),clu)
    for i in range(leni):
        use=change_list[list_gron[i]]
        list_use.append((use))
    return list_true,list_gron,list_use
if __name__ == '__main__':
    data = scio.loadmat('cora_neighboor_5_times_0.mat')
    la1=(data['label'])[0].tolist()
    print(la1)
    labels = data_loader.load_l('cora')
    labels = np.array(labels.astype("float32"))
    labels_true = np.argmax(labels, axis=1).tolist()  # 从one-hot计算真实label
    print(labels_true)
    print(acc(labels_true, la1))
