import pandas as pd
import math


def fun(sigma, mu, xi):  # 概率密度函数
    return 1 / (math.sqrt(2 * math.pi) * sigma) * math.exp(-(xi - mu) ** 2 / (2 * sigma ** 2))


def Naive_Bayers_Classifier(data, pred_data, c):
    features_list = pred_data.keys()
    c_num = data[c].value_counts().values.tolist()
    c_name = data[c].value_counts().index.tolist()
    p_c_list = []
    for i in range(len(c_num)):
        p_c_list.append((c_num[i] + 1) / (data.shape[0] + len(c_num)))

    ##每个属性估计条件概率
    die_data = data.groupby(c)
    p_y_list = []

    for i in range(len(p_c_list)):
        data1 = die_data.get_group(c_name[i])

        p_f_list = []  # 条件概率P(x_i|c)
        for feature in features_list:
            if data1[feature].dtype == object:
                f_num = data1[data1[feature] == pred_data[feature]].shape[0]
                p_f_list.append((f_num + 1) / (data1.shape[0] + len(data1[feature].value_counts().index.tolist())))
            else:
                sigma = data1[feature].std()
                mu = data1[feature].mean()
                p_f_list.append(fun(sigma, mu, pred_data[feature]))
        mul = 1
        for p_f in p_f_list:
            mul *= p_f  # P(x|c)
        mul *= p_c_list[i]
        print(str(c) + ' ' + c_name[i] + '的概率:', mul)
        p_y_list.append(mul)

    idmax = p_y_list.index(max(p_y_list))
    print('结论:' + str(c) + ' ' + c_name[idmax])


data = pd.read_csv('D:/aaa/watermelon3_0_Ch.csv')
##预测：
pred_data = {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '清脆', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑',
             '密度': 0.697, '含糖率': 0.460}

c = '好瓜'

Naive_Bayers_Classifier(data, pred_data, c)