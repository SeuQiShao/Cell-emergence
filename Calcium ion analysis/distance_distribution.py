from faulthandler import disable
import numpy as np
import pandas as pd
import os
import scipy.stats


def get_distance(data):
    distance = []
    data_grouped = data.groupby('Time')
    keys = data_grouped.groups.keys()
    for i in keys:
        temp = data_grouped.get_group(i)
        if len(temp) == 2:
            dis = np.linalg.norm(temp.iloc[0,:3]-temp.iloc[1,:3])
            distance.append(dis)
    return distance




if __name__ == '__main__':
    data_dir = '/home/shaoqi/Dataset/Cell_Contact/contact/'
    filenames = os.listdir(data_dir)
    distance = []
    for filename in filenames:
        if filename.endswith('.xls'):
            data = pd.read_excel(data_dir + filename, sheet_name='Position', header= 1)
            distance += get_distance(data)

    bins_num = 50
    data = np.array(distance)
    data_hist = np.histogram(data, bins_num)
    bins_center = (data_hist[1][1:] + data_hist[1][:-1]) / 2
    np.savetxt('distance_distribution_bins.txt',bins_center)
    np.savetxt('distance_distribution.txt',data_hist[0]/len(data))
    print(data_hist[1])

    ###置信区间
    # y.mean() + var(y)/2 + t.ppf(0.975, len(y)-1)*sqrt(var(y)/len(y) + var(y)**2 / (2(n-1)))
    y = np.log(data)
    n = len(y)
    t_score = scipy.stats.t.isf(0.05 / 2, df = (n-1) ) 
    cof_l = np.mean(y) + np.var(y)/2 - t_score * np.sqrt(np.var(y)/n + np.var(y)**2 / (2*(n-1)))
    cof_r = np.mean(y) + np.var(y)/2 + t_score * np.sqrt(np.var(y)/n + np.var(y)**2 / (2*(n-1)))
    print(np.exp(cof_l), np.exp(cof_r))

    ###置信区间
    cof_l2 = data.mean() - scipy.stats.norm.isf(0.025) * data.std() / np.sqrt(len(data))
    cof_r2 = data.mean() + scipy.stats.norm.isf(0.025) * data.std() / np.sqrt(len(data))
    print(cof_l2, cof_r2)
    np.savetxt('dis_data.txt',data)