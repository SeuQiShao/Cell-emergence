# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:41:45 2021

@author: seush
"""

import numpy as np 
import pandas as pd 
import pickle
import os
import random
from sklearn import svm
#import xgboost as xgb
from sklearn import metrics
#import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
#from cvxopt import matrix as c_matrix
#from cvxopt import solvers as c_solvers

path = ''
pos_sample = np.load(path + 'pos_sample.npy')
neg_sample = np.load(path + 'neg_sample.npy')
rate = (0.7,0.1,0.2)
epochs = 1000
pos_num = len(pos_sample)
neg_num = len(neg_sample)
pos_label = np.ones((pos_num,1))
neg_label = np.zeros((neg_num,1))
N0 = pos_num + neg_num
sample = np.vstack((pos_sample,neg_sample)).reshape(N0,-1)
#sample: sample_num*time_step*cell*position 
label = np.vstack((pos_label,neg_label))
##
label = label[sample[:,0]>0]
sample = sample[sample[:,0]>0]
N = len(sample)
#shuffle
index = list(range(N))
random.shuffle(index)
dataX = sample[index] 
dataY = label[index]
train_size = int(N*rate[0])
val_size = int(N*rate[1])
test_size = N - train_size - val_size
train_X = dataX[0:train_size]
train_Y = dataY[0:train_size]
val_X = dataX[train_size:train_size + val_size]
val_Y = dataY[train_size:train_size + val_size]
test_X = dataX[train_size + val_size:]
test_Y = dataY[train_size + val_size:]
print('train_data_length:', len(train_X))
print('test_data_length:', len(test_X))
print('val_data_length:', len(val_X))
max_value = np.max(train_X)
min_value = np.min(train_X)
#train_X = (train_X - min_value)/(max_value - min_value)
#val_X = (val_X - min_value)/(max_value - min_value)
#test_X = (test_X - min_value)/(max_value - min_value)
####SVM 0.911
model = svm.SVC(kernel='rbf', C=2.5, gamma=0.01)
model.fit(train_X, train_Y)
print('train_acc:',model.score(train_X, train_Y))
predicted= model.predict(test_X)
print('test_acc:',model.score(test_X, test_Y))
pickle.dump(model, open("svm.pickle.dat", "wb"))
loaded_model = pickle.load(open("svm.pickle.dat", "rb"))
y_pred = loaded_model.predict(test_X)
print(accuracy_score(y_pred, test_Y))
####xgboost  0.904
xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_val = xgb.DMatrix(val_X, label = val_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 8
param['silent'] = 2
param['nthread'] = 6
param['num_class'] = 2

watchlist = [(xg_train, 'train'), (xg_val, 'val')]
num_round = 1000   
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y.reshape(-1)) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
fpr, tpr, thresholds = metrics.roc_curve(test_Y.reshape(-1), pred, pos_label=1)
print('Test auc: ',metrics.auc(fpr, tpr))
print(accuracy_score(pred, test_Y))
###save
pickle.dump(bst, open("pima.pickle.dat", "wb"))
loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
y_pred = loaded_model.predict(xg_test)
