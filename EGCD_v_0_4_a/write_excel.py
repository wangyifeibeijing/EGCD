#coding=utf-8
# ==============================================================================
#
#       Filename:  demo.py
#    Description:  excel operat
#        Created:  Tue Apr 25 17:10:33 CST 2017
#         Author:  Yur
#
# ==============================================================================
import matplotlib.pyplot as plt
import xlwt
from sklearn import metrics
from basic_files import Acc_calculator
from data_set import data_loader
import scipy.io as scio
import numpy as np

import gc
def write_excel_set():
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding = 'utf-8')
    # 创建一个worksheet
    worksheet = workbook.add_sheet('result')
    worksheet_mean = workbook.add_sheet('mean')
    worksheet_var = workbook.add_sheet('var')
    name_list = [
        # 'six',
        # 'dblp',
        # 'football',
        # 'polbooks',
        # 'five_overlap',
        'cornell',
        'texas',
        'washington',
        'wisconsin',
        # 'email_EU',
        'TerrorAttack',
        # 'polblogs',
        'cora',
        'citeseer',
        'six',
        'dblp',
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

    mask_list=[
            False,
            True,
            ]
    use_h_list=[
            False,
            True,
            ]
    j=0
    for dataname in name_list:
        j = j + 1
        worksheet.write(0, j, label=dataname)# 参数对应 行, 列, 值
        worksheet_mean.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_var.write(0, j, label=dataname)  # 参数对应 行, 列, 值
    i=0
    for feat_map in feat_map_list:
        for mask in mask_list:
            for use_h in use_h_list:
                i=i+1
                name_excel = feat_map + '_mask_' + str(mask) + '_H_' + str(use_h)
                worksheet.write(i,0, label=name_excel)
                worksheet_mean.write(i,0, label=name_excel)  # 参数对应 行, 列, 值
                worksheet_var.write(i,0, label=name_excel)  # 参数对应 行, 列, 值
    j=0
    for dataname in name_list:
        j=j+1
        i=0
        for feat_map in feat_map_list:
            for mask in mask_list:
                for use_h in use_h_list:
                    i=i+1
                    try:
                        path = 'result/' + dataname
                        name_part = '_' + feat_map + '_mask' + str(mask) + '_useH' + str(use_h) + '_'
                        read_path=path+'/'+dataname+name_part+'.txt'
                        f = open(read_path, "r")  # 设置文件对象
                        re = f.read()  # 将txt文件的所有内容读入到字符串str中
                        f.close()
                        [x,y]=re.split("+", 1)
                        x = round(float(x), 4)
                        y = round(float(y), 4)
                        #name_excel = feat_map + '_mask_' + str(mask) + '_H_' + str(use_h)
                        worksheet.write(i,j, label=str(x)+'+'+str(y))
                        worksheet_mean.write(i,j, label=x)  # 参数对应 行, 列, 值
                        worksheet_var.write(i,j, label=y)  # 参数对应 行, 列, 值
                    except:
                        print(name_part)
    # 写入excel
    # 保存
    workbook.save('Excel_result.xls')


def write_excel_neigh(aim_file):
    gc.collect()
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet_nmi = workbook.add_sheet('result_nmi')
    worksheet_mean_nmi = workbook.add_sheet('mean_nmi')
    worksheet_var_nmi = workbook.add_sheet('var_nmi')
    worksheet_acc = workbook.add_sheet('result_acc')
    worksheet_mean_acc = workbook.add_sheet('mean_acc')
    worksheet_var_acc = workbook.add_sheet('var_acc')
    name_list = [

        # 'cornell',
        # 'texas',
        # 'washington',
        # 'wisconsin',

        'TerrorAttack',
        'citeseer',
        'Pubmed',
        'cora',

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
        # -1,
    ]

    j = 0
    for dataname in name_list:
        j = j + 1
        worksheet_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_mean_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_var_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_mean_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_var_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
    i = 0

    for neigh in neigh_list:
        i = i + 1
        name_excel = 'neighbor_'+ str(neigh)
        worksheet_nmi.write(i, 0, label=name_excel)
        worksheet_mean_nmi.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_var_nmi.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_acc.write(i, 0, label=name_excel)
        worksheet_mean_acc.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_var_acc.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
    j = 0
    for dataname in name_list:
        plt_acc=[]
        plt_nmi=[]
        plt_vcc=[]
        plt_vmi=[]
        max_acc_l = []
        min_acc_l = []
        max_nmi_l = []
        min_nmi_l = []
        j = j + 1
        i = 0
        labels = data_loader.load_l(dataname)

        labels = np.array(labels.astype("float32"))

        labels_true = np.argmax(labels, axis=1).tolist()  # 从one-hot计算真实label
        for neigh in neigh_list:
            i = i + 1
            loss_list=[]
            acc_list=[]
            nmi_list=[]

            name_excel = 'neighboor_' + str(neigh)
            path = aim_file+'/' + dataname
            name_path = path+'/'+dataname+'_'+name_excel+'_times_'
            for comp in range(5):
                read_pa=name_path+str(comp)+'.mat'
                print(read_pa)


                try:
                    data = scio.loadmat(read_pa)
                    #print(data)
                    l1 = (data['label'])[0].tolist()
                    gc.collect()
                    #print(l1, labels_true)
                    nmi = metrics.normalized_mutual_info_score(l1, labels_true)

                    acc = Acc_calculator.acc(labels_true, l1)
                    nmi_list.append(nmi)
                    acc_list.append(acc)
                except:
                    print(dataname+' do not have times '+str(comp))
                    print('')
            if (len(nmi_list) != 0):
                mean_nmi = round(float(np.mean(nmi_list)), 4)
                var_nmi = round(float(np.var(nmi_list)), 4)
                mean_acc = round(float(np.mean(acc_list)), 4)
                var_acc = round(float(np.var(acc_list)), 4)
                maxacc=round(float(np.max(acc_list)), 4)
                minacc = round(float(np.min(acc_list)), 4)
                maxnmi = round(float(np.max(nmi_list)), 4)
                minnmi = round(float(np.min(nmi_list)), 4)
            else:
                mean_nmi = -1
                var_nmi = -1
                mean_acc = -1
                var_acc = -1
                maxacc = 0
                minacc = 0
                maxnmi = -2
                minnmi = -2
            str_m_nmi=str(mean_nmi)
            str_v_nmi=str(var_nmi)
            excel_nmi= str_m_nmi+ '+' + str_v_nmi
            str_m_acc = str(mean_acc)
            str_v_acc = str(var_acc)
            excel_acc = str_m_acc + '+' + str_v_acc

            worksheet_nmi.write(i, j, label=excel_nmi)
            worksheet_mean_nmi.write(i, j, label=str_m_nmi)  # 参数对应 行, 列, 值
            worksheet_var_nmi.write(i, j, label=str_v_nmi)  # 参数对应 行, 列, 值
            worksheet_acc.write(i, j, label=excel_acc)
            worksheet_mean_acc.write(i, j, label=str_m_acc)  # 参数对应 行, 列, 值
            worksheet_var_acc.write(i, j, label=str_v_acc)  # 参数对应 行, 列, 值

            plt_acc.append(mean_acc)
            plt_nmi.append(mean_nmi)
            max_acc_l.append(maxacc)
            min_acc_l.append(minacc)
            max_nmi_l.append(maxnmi)
            min_nmi_l.append(minnmi)
            gc.collect()
        plt_la = 'acc'
        plt_lb = 'nmi'
        plt.plot(neigh_list,plt_acc, label=plt_la)
        plt.plot(neigh_list,plt_nmi, label=plt_lb)
        plt.fill_between(neigh_list,
                         max_acc_l,
                         min_acc_l,
                         color='b',
                         alpha=0.2)
        plt.fill_between(neigh_list,
                         max_nmi_l,
                         min_nmi_l,
                         color='r',
                         alpha=0.2)
        plt.xlabel('neighbor_num')
        plt.ylabel('acc or nmi')
        plt.legend(loc='upper right')
        plt.title('acc and nmi on ' + dataname)
        pltname = aim_file + '/' + dataname + '_differ_neigh.png'
        plt.savefig(pltname)
        plt.cla()
        gc.collect()
    # 写入excel
    # 保存
    workbook.save(aim_file+'/Excel_result_fea.xls')

def write_excel_neigh_max(aim_file):
    gc.collect()
    # 创建一个workbook 设置编码
    workbook = xlwt.Workbook(encoding='utf-8')
    # 创建一个worksheet
    worksheet_nmi = workbook.add_sheet('result_nmi')
    worksheet_mean_nmi = workbook.add_sheet('mean_nmi')
    worksheet_var_nmi = workbook.add_sheet('var_nmi')
    worksheet_acc = workbook.add_sheet('result_acc')
    worksheet_mean_acc = workbook.add_sheet('mean_acc')
    worksheet_var_acc = workbook.add_sheet('var_acc')
    name_list = [

        # 'cornell',
        # 'texas',
        # 'washington',
        # 'wisconsin',

        'TerrorAttack',
        'citeseer',
        'Pubmed',
        'cora',

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
        # -1,
    ]

    j = 0
    for dataname in name_list:
        j = j + 1
        worksheet_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_mean_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_var_nmi.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_mean_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
        worksheet_var_acc.write(0, j, label=dataname)  # 参数对应 行, 列, 值
    i = 0

    for neigh in neigh_list:
        i = i + 1
        name_excel = 'neighbor_'+ str(neigh)
        worksheet_nmi.write(i, 0, label=name_excel)
        worksheet_mean_nmi.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_var_nmi.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_acc.write(i, 0, label=name_excel)
        worksheet_mean_acc.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
        worksheet_var_acc.write(i, 0, label=name_excel)  # 参数对应 行, 列, 值
    j = 0
    for dataname in name_list:
        plt_acc=[]
        plt_nmi=[]
        plt_vcc=[]
        plt_vmi=[]
        j = j + 1
        i = 0
        labels = data_loader.load_l(dataname)

        labels = np.array(labels.astype("float32"))

        labels_true = np.argmax(labels, axis=1).tolist()  # 从one-hot计算真实label
        for neigh in neigh_list:
            i = i + 1
            loss_list=[]
            acc_list=[]
            nmi_list=[]
            name_excel = 'neighboor_' + str(neigh)
            path = aim_file+'/' + dataname
            name_path = path+'/'+dataname+'_'+name_excel+'_times_'
            for comp in range(5):
                read_pa=name_path+str(comp)+'.mat'
                print(read_pa)


                try:
                    data = scio.loadmat(read_pa)
                    gc.collect()
                    #print(data)
                    l1 = (data['label'])[0].tolist()
                    #print(l1, labels_true)
                    nmi = metrics.normalized_mutual_info_score(l1, labels_true)

                    acc = Acc_calculator.acc(labels_true, l1)
                    nmi_list.append(nmi)
                    acc_list.append(acc)
                except:
                    print(dataname+' do not have times '+str(comp))
                    print('')
            if(len(nmi_list)!=0):
                mean_nmi=round(float(np.max(nmi_list)),4)
                var_nmi=round(float(np.var(nmi_list)),4)
                mean_acc = round(float(np.max(acc_list)),4)
                var_acc = round(float(np.var(acc_list)),4)
            else:
                mean_nmi = -1
                var_nmi = -1
                mean_acc = -1
                var_acc = -1

            str_m_nmi=str(mean_nmi)
            str_v_nmi=str(var_nmi)
            excel_nmi= str_m_nmi+ '+' + str_v_nmi
            str_m_acc = str(mean_acc)
            str_v_acc = str(var_acc)
            excel_acc = str_m_acc + '+' + str_v_acc

            worksheet_nmi.write(i, j, label=excel_nmi)
            worksheet_mean_nmi.write(i, j, label=str_m_nmi)  # 参数对应 行, 列, 值
            worksheet_var_nmi.write(i, j, label=str_v_nmi)  # 参数对应 行, 列, 值
            worksheet_acc.write(i, j, label=excel_acc)
            worksheet_mean_acc.write(i, j, label=str_m_acc)  # 参数对应 行, 列, 值
            worksheet_var_acc.write(i, j, label=str_v_acc)  # 参数对应 行, 列, 值

            plt_acc.append(mean_acc)
            plt_nmi.append(mean_nmi)
            gc.collect()


    # 写入excel
    # 保存
    workbook.save(aim_file+'/Excel_result_fea_max.xls')

if __name__ == '__main__':
    aim_file='result_dot_a'
    write_excel_neigh(aim_file)
    write_excel_neigh_max(aim_file)