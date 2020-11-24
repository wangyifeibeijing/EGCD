#coding=utf-8
# ==============================================================================
#
#       Filename:  demo.py
#    Description:  excel operat
#        Created:  Tue Apr 25 17:10:33 CST 2017
#         Author:  Yur
#
# ==============================================================================

import xlwt
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