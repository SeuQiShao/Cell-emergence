from turtle import position
import numpy as np
import pandas as pd
from plot_contact import contact_judge
from plot_peak import peak_judge
import os





def generate_data_csv(data_dir, save_dir, time_interval):
    filenames = os.listdir(data_dir)
    df = pd.DataFrame()
    df['spike'] = np.arange(50)
    for filename in filenames:
        if filename.endswith(".xls"):
            print(filename)
            #position = pd.read_excel(data_dir + filename, sheet_name='Position', header= 1)
            Calcium = pd.read_excel(data_dir + filename, sheet_name='calcium', header= 1)
            peak_num = peak_judge(Calcium)
            peak_num = peak_num.tolist()
            peak = []
            for i in range(np.max(peak_num) + 1):
                peak.append(peak_num.count(i))
            peak = np.array(peak)
            peak = peak.reshape(-1)
            peak_freq = peak / peak.sum()
            df.loc[:,filename + 'Cell_num'] = pd.Series(peak)
            df.loc[:,filename + 'Cell_freq'] = pd.Series(peak_freq)
    df.to_csv(save_dir + str(time_interval)+ 'spike_num_freq.csv')


if __name__ == '__main__':
    data_dir = '/Dataset/Cell_Contact/20220516 ctrl-antag CCR2 CXCR2 anti LFA-1 xml analysis excel/'
    save_dir = '/Cell/plot_contact/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    generate_data_csv(data_dir, save_dir, 20)