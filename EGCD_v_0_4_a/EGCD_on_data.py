from data_set import data_loader
from EGCD_v_0_4_a.EGCD_model import *
import scipy.io as sio
import os
def te_EGCD(data_name,feat_map,neigh_num):
    x, a, y = data_loader.load_fast(data_name)
    [n, k] = y.shape
    path = 'result_'+feat_map+'/' + data_name
    if (os.path.exists(path) == False):
        os.mkdir(path)
    nmi_list=[]
    times=5

    for i in range(times):
        gc.collect()
        label, loss = use_EGCD(x, a, k,feat_map,neigh_num)
        aim_path = path + '/' + data_name + '_neighboor_' + str(neigh_num)+'_times_' + str(i) + '.mat'
        label1=np.array(label.detach())
        sio.savemat(aim_path,{'label':label1,'loss':loss})
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

        # 'cornell',
        # 'texas',
        # 'washington',
        # 'wisconsin',
        'cora',
        'TerrorAttack',
        'citeseer',
        'Pubmed',

    ]
    neigh_list = [
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        60,
        80,
        # -1,
    ]
    feat_list=[
        'dot',
        'dot_a',
        'dot_b',
    ]
    for dataname in name_list:
        for feat in feat_list:
            for neighbors in neigh_list:
                gc.collect()
                print('===============' + dataname + '_' + feat + '_' + str(neighbors)+ '=======================')
                te_EGCD(dataname,feat,neighbors)
