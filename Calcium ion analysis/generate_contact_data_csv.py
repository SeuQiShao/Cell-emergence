from turtle import position
import numpy as np
import pandas as pd
import os

"""
format
Cell1 Cell2 Time1 Time2 duration

"""





def contact_judge(data, time_interval):
    position_data = data.iloc[:,[0,1,2,6,7]].groupby('Time')

    for i in range(len(position_data)):
        temp = position_data.get_group(i + 1).iloc[:,[4,0,1,2]]
        if i == 0:
            data = temp
        else:
            data = pd.merge(data, temp, on='Parent', how='outer')
    data = data.fillna(0)
    data = data.values[:,1:].reshape(data.shape[0],-1,3)
    print(data.shape) #(node, time, position)
    Result = []
    for i in range(1, data.shape[0]):
        for j in range(i):
            dis = np.sqrt(np.sum((data[i] - data[j])**2, axis=1))
            start = True
            for k in range(len(dis)):
                if dis[k] < 13.77 and dis[k] > 0.01:
                    if start:
                        start = False
                        start_time = k + 1
                else:
                    if not start:
                        end_time = k
                        duration = (end_time - start_time + 1)/(60/time_interval)
                        start = True
                        Result.append([i, j, start_time, end_time, duration])
                    

    Result = np.array(Result)


    return Result








def generate_contact_data_csv(data_dir, save_dir, time_interval):
    filenames = os.listdir(data_dir)
    for filename in filenames:
        if filename.endswith(".xls"):
            print(filename)
            position = pd.read_excel(data_dir + filename, sheet_name='Position', header= 1)
            Result = contact_judge(position, time_interval)
            df = pd.DataFrame(Result, columns=['Cell1', 'Cell2', 'Time1', 'Time2', 'duration'])
            df2 = df.groupby(['Cell1', 'Cell2']).sum().iloc[:,-1]
            df2.to_csv(save_dir +'total_Contact_' + filename[:-4] + '.csv', index=True)
            df.to_csv(save_dir + 'Contact_time_' + filename[:-4] + '.csv', index=False)


if __name__ == '__main__':
    data_dir10 = '/Dataset/Cell_Contact/anti-LFA-1-CD54 xml analysis excel/'
    data_dir20 = '/Dataset/Cell_Contact/20220516 ctrl-antag CCR2 CXCR2 anti LFA-1 xml analysis excel/'
    save_dir10 = '/Cell/plot_contact/contact_data_csv_10/'
    save_dir20 = '/Cell/plot_contact/contact_data_csv_20/'
    if not os.path.exists(save_dir10):
        os.makedirs(save_dir10)
    if not os.path.exists(save_dir20):
        os.makedirs(save_dir20)
    generate_contact_data_csv(data_dir10, save_dir10, 10)
    generate_contact_data_csv(data_dir20, save_dir20, 20)
