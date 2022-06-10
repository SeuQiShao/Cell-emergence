import numpy as np
import pandas as pd
import os



"""
format:
spike cell1 contact cell2 time
1      1      True    2     3
"""

def peak_con(position, Calcium, time_plus):
    log = []
    grouped_ca = Calcium.iloc[:,[3,5]].groupby('Parent')
    keys = grouped_ca.groups.keys()
    for i in keys:
        temp =  grouped_ca.get_group(i).iloc[:,-1]
        mean =  temp.mean()
        temp[temp < 2 * mean] = 0
        temp[temp > 2 * mean] = 1
        Calcium.iloc[:,-1][Calcium['Parent'] == i] = temp

    for i in range(1, len(Calcium)):
        if Calcium.iloc[i,5] == 1:
            Calcium.iloc[i-1,5] = 0

    Result = []
    for i in range(Calcium.shape[0]):
        #pred = 0
        parent_list = []
        if Calcium.iloc[i,-1] == 1:
            contact = 0
            contact_cell = []
            parent, time = Calcium.iloc[i,3], Calcium.iloc[i,2]
            if parent not in parent_list:
                spike = 0
            spike = spike + 1
            parent_list.append(parent)
            temp = position.loc[position['Time'] == time]
            data = temp[temp['Parent'] == parent].iloc[:,[0,1,2]].values
            n_data = temp[temp['Parent'] != parent]
            ###############n_data
            for k in range(time_plus):
                temp_n = position.loc[position['Time'] == time + 1 + time_plus]
                temp_p = position.loc[position['Time'] == time - 1 - time_plus]
                data_next = temp_n[temp_n['Parent'] != parent]
                data_pre = temp_p[temp_p['Parent'] != parent]
                n_data = pd.concat([n_data, data_next])
                n_data = pd.concat([data_pre,n_data])
            ###############
            
            for i in range(len(n_data)):
                parent_i = n_data['Parent'].iloc[i]
                position_i = n_data.iloc[i,[0,1,2]].values
                dis = np.sqrt(np.sum((position_i - data)**2))
                if dis < 13.77 and dis > 0.01:
                    contact = 1
                    contact_cell.append(parent_i)
            Result.append([spike, parent, contact, set(contact_cell), time])

    Result = pd.DataFrame(Result, columns=['spike', 'parent', 'contact', 'contact_cell', 'time'])
    return Result

def generate_peak_contact(data_dir, save_dir, time_interval, time_plus = 0):
    filenames = os.listdir(data_dir)
    for filename in filenames:
        if filename.endswith(".xls"):
            print(filename)
            position = pd.read_excel(data_dir + filename, sheet_name='Position', header= 1)
            Calcium = pd.read_excel(data_dir + filename, sheet_name='calcium', header= 1)
            Result = peak_con(position, Calcium, time_plus)
            Result.to_csv(save_dir + str(time_interval) +'Contact_peak_' + filename[:-4] + '.csv', index=False)





if __name__ == '__main__':
    datadir_10 = '/Dataset/Cell_Contact/anti-LFA-1-CD54 xml analysis excel/'
    datadir_20 = '/Dataset/Cell_Contact/20220516 ctrl-antag CCR2 CXCR2 anti LFA-1 xml analysis excel/'
    save_dir20 = '/shaoqi/Cell/plot_contact/contact_spike20_0/'
    save_dir10 = '/shaoqi/Cell/plot_contact/contact_spike10_0/'
    if not os.path.exists(save_dir20):
        os.makedirs(save_dir20)
    if not os.path.exists(save_dir10):
        os.makedirs(save_dir10)
    generate_peak_contact(datadir_20, save_dir20, 20, 0)
    generate_peak_contact(datadir_10, save_dir10, 10, 0)