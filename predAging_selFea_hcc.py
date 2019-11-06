hi# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 15:50:30 2018

@author: wolf
"""
######################
####  For Main Analysis or reTest Analysis to select features
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
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
    
from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.utils import shuffle

######################
####  sum of FC/SC within and between Modules to be the feature
######################

# import ASD data

#ex_SC_path = 'F:/BrainAging/SDSU/test/Results/mat_ASD_ex_20.mat'
#in_SC_path = 'F:/BrainAging/SDSU/test/Results/mat_ASD_in_20.mat'
#ex_FC_path = 'F:/BrainAging/SDSU/FNC/SDSU_2/mat_fun_ASD_ex_20.mat'
#in_FC_path = 'F:/BrainAging/SDSU/FNC/SDSU_2/mat_fun_ASD_in_20.mat'

######################
####  step one : to test Module of prediction age using LeaveOneOut
######################


#import TD data
ex_SC_path = 'F:/BrainAging/result/NYU_SDSU_stru_ASD_ex_20.mat'
in_SC_path = 'F:/BrainAging/result/NYU_SDSU_stru_ASD_in_20.mat'
ex_FC_path = 'F:/BrainAging/result/NYU_SDSU_fun_asd_ex_20.mat'
in_FC_path = 'F:/BrainAging/result/NYU_SDSU_fun_asd_in_20.mat'

data_SC_ext = mt.loadmat(ex_SC_path)
data_SC_in = mt.loadmat(in_SC_path)
data_FC_ext = mt.loadmat(ex_FC_path)
data_FC_in = mt.loadmat(in_FC_path)

for k_ex, v_ex in data_SC_ext.items():
    c_ex = k_ex   
#    d_ex = pd.DataFrame(np.mat(v_ex))
    d_ex = np.mat(np.mat(v_ex))

for k_in,v_in in data_SC_in.items():
    c_in = k_in
    d_in = np.mat(np.mat(v_in))

for m_ex, n_ex in data_FC_ext.items():
    w_ex = m_ex
    v_ex = np.mat(np.mat(n_ex))
    
for m_in, n_in in data_FC_in.items():
    w_in = m_in
    v_in = np.mat(np.mat(n_in))
    

data = np.hstack((d_ex,d_in,v_ex,v_in))#（SC_ex,SC_in,FC_ex,FC_in）


# import real data for prediction
#data_path = 'F:/BrainAging/result_20181112/ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z.mat'

#data_path = 'F:/BrainAging/result_20190128_child_adol/res_TD/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'

# sum and consistent ex with in
#data_path = 'F:/BrainAging/result_20190423_child_adol/res_TDC/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'

# average and consistent ex with in
#data_path = 'F:/BrainAging/result_20190429_ave_child_adol/res_TD/NYU_SDSU_TD_str_fun_ex_in_20_Z_47.mat'

#data_path = 'F:/BrainAging/result_20190128_child_adol/reTest/Data_remain_afShuff/NYU_SDSU_TD_data.mat'

#data_path = 'F:/BrainAging/result_20190128_child_adol/reTest_20190222/NYU_SDSU_ASD_ID_age_cov_str_fun_ex_in_20_Z.mat'
#data_path = 'F:/BrainAging/result_20190128_child_adol/reTest_20190222/NYU_SDSU_TD_ID_age_cov_str_fun_ex_in_20_Z.mat'
# NYU
data_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_TDC/NYU_TD_str_fun_ex_in_20_Z_23.mat'
# SDSU
#data_path = 'F:/BrainAging/result_20190423_sum_child_adol/reTest/SDSU/res_TD/SDSU_TD_str_fun_ex_in_20_Z_24.mat'
# function
#data_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/Fun/res_TD/NYU_SDSU_TD_fun_ex_in_20_Z_47.mat'
# structure
#data_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/Stru/res_TD/NYU_SDSU_TD_str_ex_in_20_Z_47.mat'
# function_264
#data_path = 'F:/BrainAging/reTest_264/res_TD/NYU_SDSU_TD_fun_264_ex_in_10_47_Z.mat'

data_SC_ext = mt.loadmat(data_path)

for k_ex, v_ex in data_SC_ext.items():
    c_ex = k_ex   
#    data = pd.DataFrame( np.mat(v_ex))
    X = np.mat(np.mat(v_ex))
    
    
X_str = X[:,0:210]
X = X_str  

    
X_fun = X[:,210:420]
X = X_fun

# import age
#ag_path = 'F:/BrainAging/result_20190128_child_adol/res_TD/NYU_SDSU_TD_age_47.mat'

# sum and consistent ex with in
#ag_path = 'F:/BrainAging/result_20190423_child_adol/res_TDC/NYU_SDSU_TD_age_47.mat'

# average and consistent ex with in
#ag_path = 'F:/BrainAging/result_20190429_ave_child_adol/res_TD/NYU_SDSU_TD_age_47.mat'

# NYU
ag_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/NYU/res_TDC/NYU_TD_age_23.mat'
# SDSU
#ag_path = 'F:/BrainAging/result_20190423_sum_child_adol/reTest/SDSU/res_TD/SDSU_TD_age_24.mat'
# function
#ag_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/Fun/res_TD/NYU_SDSU_TD_age_47.mat'
# structure
#ag_path = 'F:/BrainAging/result_20190429_ave_child_adol/reTest/Stru/res_TD/NYU_SDSU_TD_age_47.mat'
# function_264
#ag_path = 'F:/BrainAging/reTest_264/res_TD/NYU_SDSU_TD_age_47.mat'

    
age = mt.loadmat(ag_path)


for k_ag, v_ag in age.items():
    c_ag = k_ag
    y = np.array(np.mat(v_ag))
    y = y.flatten()

# exclude the effect of head motion
delIDpath = 'F:/BrainAging/reTest_264/mat_delID_ASD_TD_py.mat'    
delID = mt.loadmat(delIDpath)
for k_ID, v_ID in delID.items():
    c_ID = k_ID
    dele = np.array(v_ID)
    dele = dele.flatten()

X_del = np.delete(X,dele,axis = 1)  

X = X_del

## shuffle the data for reTest
#data = np.hstack((y,X))    
data_sh = shuffle(X)
X = data_sh[0:24,1:421]
y = data_sh[0:24,0]
y = y.flatten().T

 
#delIDpath = 'F:/BrainAging/result_child_adol_20190110/delID_ASD_TD_corrHeadMnoCov_274Fro420.mat' 
#delID = mt.loadmat(delIDpath)
#for k_ID, v_ID in delID.items():
#    c_ID = k_ID
#    dele = np.array(v_ID)
#    dele = dele.flatten()
##    
#X = np.delete(X,47, axis = 0)
#y = np.delete(y,47, axis = 0)
    
#X = np.delete(X,(0,1,2,3,4,5,6,7,8,9,10),axis = 0)
#y = np.delete(y,(0,1,2,3,4,5,6,7,8,9,10),axis = 0)

def sele_fea(X,y): # X is the data; y is the age

    
    #X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel='linear')
#    estimator = SVR()
#    estimator = LinearSVR(C=1.0, epsilon=0.1,max_iter=100,
#                          tol=1e-05)
    # step: corresponds to the (integer) number of features to remove at each iteration
    # cv: 为要分成的包的总个数
    # 
    selector = RFECV(estimator, step=1, cv=2)

    selector = selector.fit(X, y)
    sel_fea = selector.transform(X) # The sel_fea is with only the selected features
    fea_num = selector.n_features_
    sel_index = selector.get_support(True) 
    sel_rank = selector.ranking_
    print("Optimal number of features : %d" % selector.n_features_)
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
#    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.ylabel("Cross validation score")
    plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)

#    np.max((cc[50:,])
#    cc = np.argmax(cc[50:,])
#    cc1 = np.sum(sel_rank <= (50+cc+1))
#    cc2 = np.where(sel_rank <= cc1)
#    cc3 = np.array(cc2).T
#    cc4 = cc3.flatten()
#    sel_index1 = cc4
    plt.show()
    return (sel_fea, fea_num, sel_index)

# ASD
    X_del = np.delete(X,12,axis = 0)
    y_del = np.delete(y,12,axis = 0)
    
sel_fea_TD, fea_num_TD, sel_fea_index_TD = sele_fea(X_del,y_del)

np.save('sel_fea_index_str_fun_TD_25.npy', sel_index)

np.save('NYU_SDSU_TD_data_k10_croVal_249.npy', X)
np.save('NYU_SDSU_TD_age_k10_croVal_249.npy', y)



