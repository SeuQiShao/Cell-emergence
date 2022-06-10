# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 15:53:52 2021

@author: seush
"""
from sklearn import svm
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import integrate
import pickle
import xgboost as xgb


##b \in 0.3-0.8, r \in 0.3-0.8 
def ETP(x,a = 1,b = 0.60,c = 0,r = 0.13):
    y = a * np.exp(-b * (x + c)) * (x + c)**(r)
    return y
###load_model
#svm_model = pickle.load(open("svm.pickle.dat", "rb"))
xgb_model = pickle.load(open("pima.pickle.dat", "rb"))
ARRAY = np.array([[0,0,0,0,0,0]])
#TEST_pred = svm_model.predict(ARRAY)
result_test = xgb.DMatrix(ARRAY)
print('xgb_test_sample_result: ', xgb_model.predict(result_test))
#print('svm_test_sample_result: ', TEST_pred)
def generate_certain_distribution_sample(a, b, c, r):
    v, err = integrate.quad(ETP, 0.01, 10, args = (a,b,c,r))
    print('normalization:', v)
    Threshold = 1000
    ####generate ETP distribution
    x = np.arange(0.01,10.,0.01)
    plt.plot(x,ETP(x,a,b,c,r)/v,color = "red")
    size = int(3e+05)
    length = 10
    z = np.random.uniform(low = 0, high = length, size = size) #q(x)
    u = np.random.uniform(low = 0, high = 1, size = size)
    qz = 1/length
    pz =  ETP(z,a,b,c,r) #(pi(x))
    M = np.max(ETP(x,a,b,c,r)) * length  #pi(x) < Mq(x)
    sample = z[pz/(M*qz) >= u]
    sample = sample[sample < Threshold]
    plt.hist(sample,bins=100, density=True, edgecolor='black') 
    plt.savefig('ETP_sample2.png')
    plt.show()
    print(np.mean(sample))
    return sample
###random
#sample2 = 10*np.random.rand(10000)

def generate_cell_simulation(a,b,c,r0,model,cell_size):
    ####init
    cell_size = cell_size
    box_range = np.array([[0,600],[0,600],[0,65]])
    time_step = 59
    rep = 10
    sample = generate_certain_distribution_sample(a, b, c, r0)
    Final = []
    C_num = []
    C_time = []
    for u in range(rep):
        ####random_walk
        Position = []
        cell_x = 600*np.random.rand(cell_size)
        cell_y = 600*np.random.rand(cell_size)
        cell_z = 65*np.random.rand(cell_size)
        Position.append([cell_x,cell_y,cell_z])
        for i in range(time_step):
            r = random.sample(list(sample),cell_size)
            #r = random.sample(list(sample2),cell_size)
            theta = np.pi*np.random.rand(cell_size)
            phi = np.pi*2*np.random.rand(cell_size)
            cell_x = cell_x + r*np.sin(theta)*np.cos(phi)
            cell_y = cell_y + r*np.sin(theta)*np.sin(phi)
            cell_z = cell_z + r*np.cos(theta)
            Position.append([cell_x,cell_y,cell_z])
        Position_sample = np.array(Position)
        ####generate dataset
        Result = []
        for k in range(time_step + 1):
            position_data = Position_sample[k].transpose(1,0)
            Dataset = []
            for i in range(1,cell_size):
                for j in range(i):
                    cell_data = np.hstack((position_data[i],position_data[j]))
                    if np.sum((cell_data[[0,1,3,4]] < 0) | (cell_data[[0,1,3,4]] > 600)):
                        cell_data = np.array([0,0,0,0,0,0])
                    if np.sum((cell_data[[2,5]] < 0) | (cell_data[[2,5]] > 65)):
                        cell_data = np.array([0,0,0,0,0,0])
                    Dataset.append(cell_data)
            Dataset = np.array(Dataset)
            if model == xgb_model:
                test_data = xgb.DMatrix(Dataset)
                pred = model.predict(test_data)
                result_path = 'Cell/'+str(b)[:5] + '_' + str(r0)[:5] +'_' +str(cell_size) +'_xgb_final_etp.npy'
            print('rep: ',u,'time_step: ',k,'Contact number:',np.sum(pred))
            Result.append(pred)
        Final.append(Result)
        Result = np.array(Result)
        Contact_number = np.sum(Result, 1)
        Contact_time = np.sum(Result, 0)
        C_num.append(Contact_number)
        C_time.append(Contact_time)
        #file_name = str(cell_size) + '_' + str(time_step) + '_' + str(u) + '_etp.txt'
        #np.savetxt(file_name,Result)
    Final = np.array(Final)
    np.save(result_path,Final)

####simulation
# a = 1
# c = 0
#generate_cell_simulation(a = 1,b = 0.80,c = 0,r0 = 0.47, model = model)
# for b in 0.1 * np.array(range(2,31)):
#     for r in 0.1 * np.array(range(2,19)):
#         print('b = ',b, 'r = ',r)
#         generate_cell_simulation(a,b,c,r,xgb_model)
# b = 0.60
# r = -0.13
# generate_cell_simulation(a,b,c,r,xgb_model)
    

# vehicle = [0.364867, 0.512133, -0.32243, 0.2686]
# CXCR2 = [0.5533, 0.741733, -0.4, 0.067093]
# CCR2 = [0.630533, 0.846033, -0.4, 0.073123]
# CX3CR1 = [0.4149, 0.593667, -0.2601, 0.44143]
# CXCR4 = [0.3995, 0.556033, -0.35967, 0.15782]
# CCR1 = [0.400133, 0.5699, -0.32447,0.261803]
# CCR5 = [0.407867, 0.561, -0.36523, 0.15524]
# true_para = [vehicle, CXCR2, CCR2, CX3CR1, CXCR4, CCR1, CCR5]
# for i in true_para:
#     a,b,c,r = i
#     generate_cell_simulation(a,b,c,r,xgb_model)
# a,b,c,r = true_para[0]
# generate_cell_simulation(a,b,c,r,xgb_model)
cell_size = [100, 150, 200, 250, 300]
for i in cell_size:
    generate_cell_simulation(a = 1,b = 0.60,c = 0,r0 = -0.13, model = xgb_model, cell_size = i)