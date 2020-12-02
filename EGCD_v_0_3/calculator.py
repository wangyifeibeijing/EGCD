import scipy.io as scio
from sklearn import metrics
import data_loader
import numpy as np

def calculate_one(name,labels):
    nmi_ram=[]
    for i in range(5):
        path = 'result/no_con/'+name+'_no_con/result_'+name+'_no_con'+str(i)+'.mat'
        data = scio.loadmat(path)
        labels = np.array(labels.astype("float32"))
        labels_true = np.argmax(labels, axis=1)
        y=data['mi'][0]
        nmi=metrics.normalized_mutual_info_score(y, labels_true)
        nmi_ram.append(nmi)
    meani=np.mean(nmi_ram)
    vari=np.var(nmi_ram)
    print('nmi on '+name+':'+'mean: '+str(meani)+' var: '+str(vari))
def calculate_con(name,labels):
    nmi_ram=[]
    for i in range(5):
        path = 'result/constrained/'+name+'_con/result_'+name+'__con'+str(i)+'.mat'
        data = scio.loadmat(path)
        labels = np.array(labels.astype("float32"))
        labels_true = np.argmax(labels, axis=1)
        y=data['mi'][0]
        nmi=metrics.normalized_mutual_info_score(y, labels_true)
        nmi_ram.append(nmi)
    meani=np.mean(nmi_ram)
    vari=np.var(nmi_ram)
    print('nmi on '+name+':'+'mean: '+str(meani)+' var: '+str(vari))
def cal_no():
    print('no constraint')
    # dataname='citeseer'
    # features, Amatrix, labels = data_loader.load_citeseer()

    dataname = 'cora'
    features, Amatrix, labels = data_loader.load_cora()
    calculate_one(dataname, labels)

    dataname = 'cornell'
    features, Amatrix, labels = data_loader.load_cornell()
    calculate_one(dataname, labels)

    dataname = 'foot'
    features, Amatrix, labels = data_loader.load_football()
    calculate_one(dataname, labels)

    dataname = 'pollogs'
    features, Amatrix, labels = data_loader.load_polblogs()
    calculate_one(dataname, labels)

    dataname = 'texas'
    features, Amatrix, labels = data_loader.load_texas()
    calculate_one(dataname, labels)

    dataname = 'washington'
    features, Amatrix, labels = data_loader.load_washington()
    calculate_one(dataname, labels)

    dataname = 'wisconsin'
    features, Amatrix, labels = data_loader.load_wisconsin()
    calculate_one(dataname, labels)
def cal_co():
    print('constrainted')
    # dataname='citeseer'
    # features, Amatrix, labels = data_loader.load_citeseer()

    dataname = 'cora'
    features, Amatrix, labels = data_loader.load_cora()
    calculate_con(dataname, labels)

    dataname = 'cornell'
    features, Amatrix, labels = data_loader.load_cornell()
    calculate_con(dataname, labels)

    dataname = 'foot'
    features, Amatrix, labels = data_loader.load_football()
    calculate_con(dataname, labels)

    dataname = 'pollogs'
    features, Amatrix, labels = data_loader.load_polblogs()
    calculate_con(dataname, labels)

    dataname = 'texas'
    features, Amatrix, labels = data_loader.load_texas()
    calculate_con(dataname, labels)

    dataname = 'washington'
    features, Amatrix, labels = data_loader.load_washington()
    calculate_con(dataname, labels)

    dataname = 'wisconsin'
    features, Amatrix, labels = data_loader.load_wisconsin()
    calculate_con(dataname, labels)

if __name__ == '__main__':
    cal_co()
    cal_no()