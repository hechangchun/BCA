# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:23:31 2018

@author: wolf
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:25:14 2018

@author: wolf
"""
######################
####  For Main Analysis to predict
######################

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import mat4py as mt
import pickle as pk
import h5py as hp
import scipy as sci


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from liblinearutil import *
from svmutil import *
from svm import *
from numpy import mat
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
    
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.utils import shuffle

######################
####  sum of FC/SC within rather than between Modules to be the feature
######################
##### For real prediction
# import ASD data
#data_path_asd = 'F:/BrainAging/result_20190128_child_adol/res_ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z_57.mat'
#ag_path_asd = 'F:/BrainAging/result_20190128_child_adol/res_ASD/NYU_SDSU_ASD_age_57.mat'

# sum and consistent ex with in
#data_path_asd = 'F:/BrainAging/result_20190423_child_adol/res_ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z_57.mat'
#ag_path_asd = 'F:/BrainAging/result_20190423_child_adol/res_ASD/NYU_SDSU_ASD_age_57.mat'

# average and consistent ex with in
#data_path_asd = 'F:/BrainAging/result_20190429_ave_child_adol/res_ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z_52.mat'
#ag_path_asd = 'F:/BrainAging/result_20190429_ave_child_adol/res_ASD/NYU_SDSU_ASD_age_52.mat'

# NYU for retest
data_path_asd = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_ASD/NYU_ASD_str_fun_ex_in_20_Z_22.mat'
ag_path_asd = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_ASD/NYU_ASD_age_22.mat'

#data_path_asd = 'F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/NYU_SDSU_ASD_data.mat'
#ag_path_asd = 'F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/NYU_SDSU_ASD_age.mat'
data_asd_ext = mt.loadmat(data_path_asd)
for k_ex_asd, v_ex_asd in data_asd_ext.items():
    c_ex_asd = k_ex_asd   
#    data = pd.DataFrame( np.mat(v_ex))
    X_asd = np.mat(v_ex_asd)
#    X1 = float(X - np.tile(np.mean(X, axis = 0),(46,1))) / np.tile(np.std(X, axis = 0),(46,1))
    
age_asd = mt.loadmat(ag_path_asd)

for k_ag_asd, v_ag_asd in age_asd.items():
    c_ag_asd = k_ag_asd
    y_asd = np.array(v_ag_asd)
    y_asd = y_asd.flatten()

# only explore child and adolescence  
#X_asd = np.delete(X_asd,(19,20),axis = 0)
#y_asd = np.delete(y_asd,(19,20),axis = 0)
 
# import TD data
# data_path_TD = 'F:/BrainAging/result_20190128_child_adol/res_TD/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'
# ag_path_TD = 'F:/BrainAging/result_20190128_child_adol/res_TD/NYU_SDSU_TD_age_47.mat'
# 
 # sum and consistent ex with in
#data_path_TD = 'F:/BrainAging/result_20190423_child_adol/res_TDC/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'
#ag_path_TD = 'F:/BrainAging/result_20190423_child_adol/res_TDC/NYU_SDSU_TD_age_47.mat'

# average and consistent ex with in
#data_path_TD = 'F:/BrainAging/result_20190429_ave_child_adol/res_TD/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'
#ag_path_TD = 'F:/BrainAging/result_20190429_ave_child_adol/res_TD/NYU_SDSU_TD_age_47.mat'

# NYU for retest
data_path_TD = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_TDC/NYU_TD_str_fun_ex_in_20_Z_23.mat'
ag_path_TD = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_TDC/NYU_TD_age_23.mat'
 
#  data_path_TD = 'F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/NYU_SDSU_TD_data.mat'
# ag_path_TD = 'F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/NYU_SDSU_TD_age.mat'

data_TD_ext = mt.loadmat(data_path_TD)
for k_ex_TD, v_ex_TD in data_TD_ext.items():
    c_ex_TD = k_ex_TD   
#    data = pd.DataFrame( np.mat(v_ex))
    X_td  = np.mat(v_ex_TD)
#    X1 = float(X - np.tile(np.mean(X, axis = 0),(46,1))) / np.tile(np.std(X, axis = 0),(46,1))
    
age_TD = mt.loadmat(ag_path_TD)

for k_ag_TD, v_ag_TD in age_TD.items():
    c_ag_TD = k_ag_TD
    y_TD = np.array(v_ag_TD)
    y_TD = y_TD.flatten()
    
    
# structure   
X_str_asd = X_asd[:,0:210]
X_asd = X_str_asd  

X_str_td = X_td[:,0:210]
X_td = X_str_td 

sel_fea_index_TD = np.load('F:/BrainAging/result_20190128_child_adol/reTest_20190304/Stru/sel_fea_index_str_fun_TD_96.npy')

# Function     
X_fun_asd = X_asd[:,210:420]
X_asd = X_fun_asd

X_fun_td = X_td[:,210:420]
X_td = X_fun_td
    
# main analysis    
#sel_fea_index_TD = np.load('F:/BrainAging/result_20190128_child_adol/sel_fea_index_str_fun_TD_12.npy')
#inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,9,axis = 0) # union between ASD and TD correlated with motion
    
# consistent ex with in
#sel_fea_index_TD = np.load('F:/BrainAging/result_20190429_ave_child_adol/sel_fea_index_str_fun_TD_131.npy')
#inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,9,axis = 0) # union between ASD and TD correlated with motion

# NYU for retest
sel_fea_index_TD = np.load('F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/sel_fea_index_str_fun_TD_25.npy')

#reTest analysis (k = 10 cross validation)
#sel_fea_index_TD = np.load('F:/BrainAging/result_20190128_child_adol/reTest/k_10_crossValid/sel_fea_index_str_fun_k10_croVal_TD_249.npy')

delIDpath = 'F:/BrainAging/result_20190423_child_adol/mat_delID_ASD_TD.mat'    
delID = mt.loadmat(delIDpath)
for k_ID, v_ID in delID.items():
    c_ID = k_ID
    dele = np.array(v_ID)
    dele = dele.flatten()

inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,dele,axis = 0) # union between ASD and TD correlated with motion
#inter_fea_index_TD_delMot = np.load('F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/sel_fea_index_str_fun_TD_73.npy')

X_asd_del = X_asd[:,sel_fea_index_TD]
X_td_del = X_td[:,sel_fea_index_TD]

#X_td = np.delete(X_td,(0,1,2,3,4,5,6,7,8,9,10),axis = 0)
#y_TD = np.delete(y_TD,(0,1,2,3,4,5,6,7,8,9,10),axis = 0)

# delIDpath = 'F:/BrainAging/result_child_adol_20190110/delID_corrHeadMnoCov_TD_ASD_82Fro165.mat'    
#delID = mt.loadmat(delIDpath)
#for k_ID, v_ID in delID.items():
#    c_ID = k_ID
#    dele = np.array(v_ID)
#    dele = dele.flatten()

###### For reTest    
## shuffle the ASD data for reTest
    
#data_path_asd = 'F:/BrainAging/result_20190128_child_adol/res_ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z.mat'
#ag_path_asd = 'F:/BrainAging/result_20190128_child_adol/res_ASD/NYU_SDSU_ASD_age.mat'
#data_asd_ext = mt.loadmat(data_path_asd)
#for k_ex_asd, v_ex_asd in data_asd_ext.items():
#    c_ex_asd = k_ex_asd   
##    data = pd.DataFrame( np.mat(v_ex))
#    X_asd = np.mat(v_ex_asd)
##    X1 = float(X - np.tile(np.mean(X, axis = 0),(46,1))) / np.tile(np.std(X, axis = 0),(46,1))
#    
#age_asd = mt.loadmat(ag_path_asd)
#
#for k_ag_asd, v_ag_asd in age_asd.items():
#    c_ag_asd = k_ag_asd
#    y_asd = np.mat(v_ag_asd)
#data_asd = np.hstack((y_asd,X_asd))    
#data_asd_sh = shuffle(data_asd)
#X_asd = data_asd_sh[0:29,1:421]
#y_asd = data_asd_sh[0:29,0]
#np.save('NYU_SDSU_ASD_data_shuffle_29.npy', X_asd)
#np.save('NYU_SDSU_ASD_age_shuffle_29.npy', y_asd)

## shuffle the ASD data for reTest from compared by above code
X_asd = np.mat(np.load('F:/BrainAging/result_20190128_child_adol/reTest/NYU_SDSU_ASD_data_shuffle_29.npy'))
y_asd = np.load('F:/BrainAging/result_20190128_child_adol/reTest/NYU_SDSU_ASD_age_shuffle_29.npy')
y_asd = y_asd.flatten()

## shuffle the TD data for reTest from the feature selection
X_td = np.mat(np.load('F:/BrainAging/result_20190128_child_adol/reTest/NYU_SDSU_TD_data_shuffle_24.npy'))
y_TD = np.load('F:/BrainAging/result_20190128_child_adol/reTest/NYU_SDSU_TD_age_shuffle_24.npy')
y_TD = y_TD.flatten()

## feature label from shuffle the TD data
sel_fea_index_TD = np.load('F:/BrainAging/result_20190128_child_adol/reTest/sel_fea_index_str_fun_TD_20.npy')
inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,(18,19),axis = 0) # union between ASD and TD correlated with motion

X_asd_del = X_asd[:,inter_fea_index_TD_delMot]
X_td_del = X_td[:,inter_fea_index_TD_delMot]

#X_asd = np.delete(X_asd_del,dele,axis = 1)   
#X_td = np.delete(X_td_del,dele, axis = 1)

# only explore child and adolescence     
#X_td = np.delete(X_td,12,axis = 0)
#y_TD = np.delete(y_TD,12,axis = 0)

####### the first part 
    
#  obtain the index of feature 
def sele_fea(X,y): # X is the data; y is the age

    
    #X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    # step: corresponds to the (integer) number of features to remove at each iteration
    # cv: 为要分成的包的总个数
#    estimator = LinearSVR(C=1.0, epsilon=0.0,random_state=0, tol=1e-05, verbose=0)
    selector = RFECV(estimator, step=1, cv=2) 
    selector = selector.fit(X, y)
    sel_fea = selector.transform(X) # The sel_fea is with only the selected features
    fea_num = selector.n_features_
    sel_index = selector.get_support(True)
    print("Optimal number of features : %d" % selector.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
#    plt.annotate('',xy = (np.argmax(selector.grid_scores_) + 1,selector.grid_scores_[np.argmax(selector.grid_scores_,)]), xytext = (np.argmin(results[1:80]),3+results[np.argmin(results[1:80])]), arrowprops=dict(facecolor='red',shrink=20))
#    plt.text(np.argmin(results[1:80])-6,(results[np.argmin(results[1:80])]-1),r'MAE = %.2f'%results[np.argmin(results[1:80])],fontsize = 10)
#    plt.text(np.argmin(results[1:80])-5,(results[np.argmin(results[1:80])]+3.5),r'K = %d'%np.argmin(results[1:80]),fontsize = 10)
    plt.xlabel("Number of features selected (K)")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.grid_scores_) + 1),  selector.grid_scores_)
#    plt.savefig('F:/BrainAging/SDSU/test/Results/panel5_TD_mae.png',format = 'png',dpi = 1000)
    plt.show()
    return (sel_fea, fea_num, sel_index)

# ASD
sel_fea_asd, fea_num_asd, sel_fea_index_asd = sele_fea(X,y)
np.save('sel_fea_index_ASD_max_min.npy', sel_fea_index_asd)
# TD
#sel_fea_TD, fea_num_TD, sel_fea_index_TD = sele_fea(X,y)
#np.save('sel_fea_index_TD.npy', sel_fea_index_TD)

######## the second part

# ineraction and union between ASD and TD
#sel_fea_index_TD = np.load('sel_fea_index_TD_78.npy')
#inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,(2,7,12,13,14,17,18,19,33,42,44,45,
#                                                        46,48,59,61,62,67,68,69,70,71,72,73,76),axis = 0) # union between ASD and TD correlated with motion

#sel_fea_index_TD = np.load('sel_fea_index_str_fun_TD_165.npy')
#inter_fea_index_TD_delMot = np.delete(sel_fea_index_TD,(0,5,9,13,16,18,	19,	21,	22,	24,	26,	27,	30,	35,41,55,	
#                                                        57, 59,66,68,70,71,74,77,78,79,87	,88	,89,92,93,	
#                                                        96,97,99,101,102,105,106,108,109,110,111,112,113,114,115,
#                                                        116,117,118,119,120,121,122,123,124,125,126,127,128,129,131,132,
#                                                        133,134,136,139,140,141,144,145,146,147,148,149,152,153,156,157,160,162,163,164,
#                                                         ),axis = 0) # union between ASD and TD correlated with motion

#sel_fea_index_ASD = np.load('sel_fea_index_ASD_172.npy')

# reChoose feature after correlation with head motion without regressing site and sex

#sel_fea_index_TD_new = np.delete(sel_fea_index_TD,(22,23,24),axis = 0)
#sel_fea_index_ASD_new = np.delete(sel_fea_index_ASD,0,axis = 0)

# reChoose feature after correlation with head motion with regressing site and sex
#sel_fea_index_TD_new = np.delete(sel_fea_index_TD,(24),axis = 0)
#sel_fea_index_ASD_new = np.delete(sel_fea_index_ASD,(27,30,31),axis = 0)

# intersection
# inter_fea_index = list(set(sel_fea_index_TD).intersection(set(sel_fea_index_ASD)))
# #feature = 5，10 were related with motion (stru & fun)
## inter_fea_index_delMot = np.delete(inter_fea_index,(5,10),axis = 0) 
## feature = 1，2, 5 were related with motion (stru)
#  inter_fea_index_delMot = np.delete(inter_fea_index,(5),axis = 0) #5 denote the index of array
 
 
# np.save('inter_fea_index.npy', inter_fea_index)
#union
# union_fea_index = list(set(sel_fea_index_TD).union(set(sel_fea_index_ASD))) # 交集


# start predition for age
#  intersection
# X_asd = X_asd[:,sel_fea_index_TD]
 X_asd = X_asd_del
 
# X_td = X_td[:,sel_fea_index_TD]
 X_td = X_td_del
 
# ASD prediction
# X_asd = np.delete(X_asd,1,axis = 1)
# X = X[:,inter_fea_index_delMot]
 
  linear
svr = SVR(kernel='linear') 
 
# rbf
#svr = SVR(kernel='poly') 
# linearSVR
#svr = LinearSVR(random_state = 0, tol = 1e-4)

#svr = NuSVR(kernel='linear', C=0.1, nu=0.5)

y_pred_ASD = np.zeros([X_asd.shape[0],1])
for j in range(X_asd.shape[0]):
    print(j)
    svr.fit(X_td,y_TD) 
    weight = svr.coef_.T
    y_pred_ASD[j] = svr.predict(X_asd[j])
    j = j + 1     
#    r2_val_ASD = metrics.mean_absolute_error(y[0:59,],  y_pred[0:59,])
#    r2_val_HC = metrics.mean_absolute_error(y[59:107,],  y_pred[59:107,]) 

# TD prediction 
loo = LeaveOneOut()
s = 0
# linear
svr = SVR(kernel='linear') 
# rbf
#svr = SVR(kernel='rbf') 
#svr = LinearSVR(random_state = 0, tol = 1e-4)

#svr = NuSVR(kernel='linear', C=0.1, nu=0.5)

y_pred_TD = np.zeros([X_td.shape[0],1])
weight_TD = np.zeros([X_td.shape[0],X_td.shape[1]])

for train, test in loo.split(X_td):
    print(s)
    svr.fit(X_td[train], y_TD[train]) 
    weight_TD[s] = svr.coef_
    y_pred_TD[s] = svr.predict(X_td[test])
    s = s + 1     
sum_w = np.sum(weight_TD,axis = 0);
sum_w1 = np.abs(np.sum(weight_TD,axis = 0));


#predictor for age
y_pred_ASD, r2_val_ASD = pred_age(X, y)

mm = metrics.mean_absolute_error(y,y_pred_ASD)

import scipy
y_pred = y_pred.flatten()
print(scipy.stats.pearsonr(y,  y_pred))
r_ASD, p_ASD = scipy.stats.pearsonr(y[0:59,],  y_pred[0:59,])
r_TD, p_TD = scipy.stats.pearsonr(y[59:107,],  y_pred[59:107,])
r, p = scipy.stats.pearsonr(y, y_pred)

# save
np.save('y_pred_ASD.npy',y_pred_ASD)
np.save('MAE_ASD', r2_val_ASD)

    
    
    