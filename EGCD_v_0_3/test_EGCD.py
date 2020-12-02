from EGCD_v_0_3 import EGCD_similarity
from EGCD_v_0_3 import data_loader
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from EGCD_v_0_3 import Acc_calculator
import scipy.io as io
import os
def te_EGCD(dataname,feat_map,neighbors,mask,use_h=False):
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
    if(neighbors!=-1):
        path = 'result/' + dataname
    else:
        path = 'result_nofe/' + dataname

    if (os.path.exists(path) == False):
        os.mkdir(path)
    nmi_list=[]
    acc_list=[]
    name_part = '_' + 'neighbor_num='+str(neighbors)

    if(size<1000):
        nmi_times=3
    elif(size<3000):
        nmi_times = 3
    else:
        nmi_times = 2
    for i in range(nmi_times):
            mi,loss_ram,y_original,H=EGCD_similarity.use_EGCD(features,Amatrix,labels,feat_map,neighbors,mask,printloss=False)#调用EGCD
            #绘制loss图
            result_mi = np.array(mi.detach())
            result_loss_ram = np.array(loss_ram)
            result_y_original = np.array(y_original.detach())
            result_H= np.array(H.detach())
            name=dataname+name_part+'times'+str(i)+'.mat'
            file_path=path+'/'+name
            io.savemat(file_path, {'mi': result_mi,
                              'loss_ram':result_loss_ram,
                              'y_original':result_y_original,
                              'H':result_H,
                              })
            A = np.array(mi)  # 将输出label转nparray
            nmi = metrics.normalized_mutual_info_score(A, labels_true)
            acc = Acc_calculator.use_acc(labels_true,A)
            # print(A)
            # print('proposed on '+dataname+'_'+feat_map)
            prt=str(nmi)+' and '+str(acc)
            print(prt)
            nmi_list.append(nmi)
            acc_list.append(acc)
            loss_fin=(result_loss_ram[-1]).detach()
            loss_final=loss_fin.float()
            nmi=round(float(nmi), 4)
            acc = round(float(acc), 4)
            loss_final = round(float(loss_final), 4)
            plt_la=' l:'+str(loss_final)+' nmi:'+str(nmi)+' acc:'+str(acc)
            plt.plot(loss_ram,label=plt_la)

            plt.xlabel('times')
            plt.ylabel('loss value')
            plt.legend(loc='upper right')
    meani = round(float(np.mean(nmi_list)),4)
    vari = round(float(np.var(nmi_list)),4)
    acci = round(float(np.mean(acc_list)), 4)
    vcci = round(float(np.var(acc_list)), 4)
    logfile = path + '/' + dataname + name_part + '.txt'
    file = open(logfile, 'w')
    nmi_title=str(meani) + '+' + str(vari)
    acc_title = str(acci) + '+' + str(vcci)
    file.write(str(meani) + '+' + str(vari))
    file.close()
    plt.title('Loss value on ' + dataname + name_part + '\n' +'nmi:'+nmi_title+' acc:'+acc_title)
    pltname = path + '/' + dataname + name_part+ '.png'
    plt.savefig(pltname)
    plt.cla()

if __name__ == '__main__':
    name_list = [
        'cornell',
        'texas',
        'washington',
        'wisconsin',
        'TerrorAttack',
        'cora',
        'citeseer',
    ]
    nofe_name_list = [
        'six',
        'football',
        'polbooks',
        'email_EU',
        'polblogs',

    ]
    feat_map_list=[
        #'rbf',
        'dot',
        #'none',
    ]
    neigh_list=[
        #5,
        40,
        35,
        30,
        25,
        20,
        15,
        10,
    ]
    nofe_name_list = [

    ]
    name_list =  [

        'TerrorAttack',

        'Pubmed_small',
        'cora',
        'citeseer',
    ]
    for dataname in nofe_name_list:
        feat_map = 'none'
        print('===============' + dataname + '_' + str(feat_map) + '=======================')
        te_EGCD(dataname, feat_map, -1, False, False)
    for dataname in name_list:
        for feat_map in feat_map_list:
            for neighbors in neigh_list:
                print('==============='+dataname+'_'+str(feat_map)+'_' + str(neighbors)+'=======================')
                te_EGCD(dataname, feat_map,neighbors,False,False)
