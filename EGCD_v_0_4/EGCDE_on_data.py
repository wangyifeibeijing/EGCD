from data_set import data_loader
from EGCD_v_0_4.EGCDE_model import *
import scipy.io as sio
import os
def te_EGCD(data_name,feat_map,neigh_num):
    x, a, y = data_loader.load_fast(data_name)
    [n, k] = y.shape
    path = 'result/' + data_name
    if (os.path.exists(path) == False):
        os.mkdir(path)
    nmi_list=[]
    times=3
    if(n>=2000):
        times=2
    for i in range(times):
        gc.collect()
        label, loss = use_EGCDE(x, a, k,feat_map,neigh_num)
        aim_path = path + '/' + data_name + '_neighboor_' + str(neigh_num)+'_times_' + str(i) + '.mat'
        sio.savemat(aim_path,{'label':label,'loss':loss})
        labels_true = np.argmax(y, axis=1)
        nmi = metrics.normalized_mutual_info_score(label, labels_true)
        print((nmi))
        nmi_list.append((nmi))
        loss_fin = (loss[-1])
        loss_final = loss_fin
        nmi = round(float(nmi), 4)
        loss_final = round(float(loss_final), 4)
        plt_la = ' l:' + str(loss_final) + ' nmi:' + str(nmi)
        plt.plot(loss, label=plt_la)
        plt.xlabel('times')
        plt.ylabel('loss value')
        plt.legend(loc='upper right')
    meani = round(float(np.mean(nmi_list)), 4)
    vari = round(float(np.var(nmi_list)), 4)
    nmi_title=str(meani)+'+'+str(vari)
    plt.title('Loss value on ' + data_name +  '_neighboor_' + str(neigh_num) + '\n' + 'nmi:' + nmi_title)
    pltname = path + '/' + data_name + '_neighboor_' + str(neigh_num) + '.png'
    plt.savefig(pltname)
    plt.cla()
    gc.collect()
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
    neigh_list = [

        10,
        20,
        30,
        40,
        60,
        80,
        100,
        -1,
    ]
    for dataname in name_list:
        for neighbors in neigh_list:
            gc.collect()
            print('===============' + dataname + '_' + str(neighbors)+ '=======================')
            te_EGCD(dataname,'dot',neighbors)
