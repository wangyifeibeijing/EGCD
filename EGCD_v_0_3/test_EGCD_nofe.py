import EGCD_similarity
import data_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import scipy.io as io
import os
def te_EGCD(dataname,feat_map,mask,use_h=False):
    #加载DATASET
    features,Amatrix,labels=data_loader.load_which(dataname)
    features=np.array(features.astype("float32"))
    Amatrix=np.array(Amatrix.astype("float32"))
    labels=np.array(labels.astype("float32"))
    [size, d] = features.shape
    labels_true = np.argmax(labels, axis=1)#从one-hot计算真实label
    if(size==d):
        feat_map='none'
        mask=False
    #print(labels_true)
    path = 'result/' + dataname
    if (os.path.exists(path) == False):
        os.mkdir(path)
    nmi_list=[]
    name_part = '_' + feat_map + '_mask' + str(mask) + '_useH' + str(use_h) + '_'

    if(size<1000):
        nmi_times=3
    elif(size<2500):
        nmi_times = 2
    else:
        nmi_times = 1
    for i in range(nmi_times):
            #print(i)
            if(use_h):
                mi, loss_ram, y_original, H = EGCD_similarity.use_EGCD_H(features, Amatrix, labels, feat_map, mask,
                                                                       printloss=False)  # 调用EGCD_H
            else:
                mi,loss_ram,y_original,H=EGCD_similarity.use_EGCD(features,Amatrix,labels,feat_map,mask,printloss=False)#调用EGCD
            #绘制loss图
            result_mi = np.array(mi.detach())
            result_loss_ram = np.array(loss_ram)
            result_y_original = np.array(y_original.detach())
            result_H= np.array(H.detach())
            name=dataname+name_part+str(i)+'.mat'
            file_path=path+'/'+name
            io.savemat(file_path, {'mi': result_mi,
                              'loss_ram':result_loss_ram,
                              'y_original':result_y_original,
                              'H':result_H,
                              })

            plt.plot(loss_ram)
            plt.title('Loss value on '+dataname+name_part)
            plt.xlabel('times')
            plt.ylabel('loss value')
            pltname=path+'/'+dataname+name_part+'_'+str(i)+'.png'
            plt.savefig(pltname)
            A=np.array(mi)#将输出label转nparray
            nmi=metrics.normalized_mutual_info_score(A, labels_true)
            #print(A)
            #print('proposed on '+dataname+'_'+feat_map)
            print(nmi)
            nmi_list.append(nmi)
    plt.cla()
    meani = np.mean(nmi_list)
    vari = np.var(nmi_list)
    logfile=path+'/'+dataname+name_part+'.txt'
    file = open(logfile, 'w')

    file.write(str(meani)+'+'+str(vari))
    file.close()
if __name__ == '__main__':
    name_list = [
        #'six',
        #'dblp',
        #'football',
        #'polbooks',
        #'five_overlap',
        'cornell',
        'texas',
        'washington',
        'wisconsin',
        #'email_EU',
        'TerrorAttack',
        #'polblogs',
        'cora',
        'citeseer',

    ]
    nofe_name_list = [
        'six',
        #'dblp',
        'football',
        'polbooks',
        # 'five_overlap',

        'email_EU',

        'polblogs',

    ]
    feat_map_list=[
        'rbf',
        'dot',
        'none',
    ]


    use_h_list=[
        False,
        True,
        ]
    testname_list = [

        'six',

    ]

    for dataname in nofe_name_list:
        feat_map = 'none'
        mask = False
        for use_h in use_h_list:
            print('==============='+dataname+'_'+str(feat_map)+'_' + str(mask)+'_' +str(use_h) + '=======================')
            te_EGCD(dataname, feat_map,mask,use_h)
