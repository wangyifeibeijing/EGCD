import numpy as np
import random
from EGCD_v_0_3 import data_loader
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
import scipy.io as scio
import matplotlib.pyplot as plt

# 需要導入模塊: from sklearn.utils import linear_assignment_ [as 別名]
# 或者: from sklearn.utils.linear_assignment_ import linear_assignment [as 別名]
def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

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
    data = scio.loadmat('result_1128/washington/washington_neighbor_num=10times0.mat')
    mi=data['mi']
    print(mi)
    data = scio.loadmat('result_1128/washington/washington_neighbor_num=10times1.mat')
    mi = data['mi']
    print(mi)
    #example#
    '''
    features, Amatrix, labels = data_loader.load_which('cora')
    features = np.array(features.astype("float32"))
    Amatrix = np.array(Amatrix.astype("float32"))
    labels = np.array(labels.astype("float32"))
    [size, d] = features.shape
    l2 = np.argmax(labels, axis=1).tolist()
    path= 'result/cora/cora_neighbor_num=25times1.mat'
    data=scio.loadmat(path)
    l1=(data['mi'])[0].tolist()
    
    
    print(l1)
    print(l2)
    W=use_acc(l1,l2)
    print(W)
    '''

    '''
    check_time=100
    for i in range(check_time):
        len = random.sample(range(5, 500), 1)[0]
        clu = random.sample(range(2, 40), 1)[0]
        k_t = random.sample(range(0, len-1), 1)[0]
        lt, lg, lu = test_list_gene(k_t, len, clu)
        ischageright = use_acc(lg, lu)
        acc1 = use_acc(lt, lg)
        acc2 = use_acc(lt, lu)
        if(ischageright<1):
            print('lg-lu wrong')
            print(lg)
            print(lu)
        elif(acc1!=acc2):
            print('acc wrong')
            print(lt)
            print(lg)
            print(lu)

     '''


