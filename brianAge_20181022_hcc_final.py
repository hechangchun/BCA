# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 10:25:14 2018

@author: wolf
"""

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

######################
####  sum of FC/SC within rather than between Modules to be the feature
######################

# import ASD data
# data_path = 'F:/BrainAging/result_20181112/ASD/NYU_SDSU_ASD_str_fun_ex_in_20_Z.mat'
# ag_path = 'F:/BrainAging/result_20181112/ASD/NYU_SDSU_ASD_age.mat'
 
 # import TD data
#  data_path = 'F:/BrainAging/result_20181112/TD/NYU_SDSU_TD_str_fun_ex_in_20_Z.mat'
# ag_path = 'F:/BrainAging/result_20181112/TD/NYU_SDSU_TD_age.mat'
# import ASD_TD data
 data_path = 'F:/BrainAging/result_20181112/TD/NYU_SDSU_TD_str_fun_ex_in_20_Z.mat'
 ag_path = 'F:/BrainAging/result_20181112/TD/NYU_SDSU_TD_age.mat'

data_SC_ext = mt.loadmat(data_path)
for k_ex, v_ex in data_SC_ext.items():
    c_ex = k_ex   
#    data = pd.DataFrame( np.mat(v_ex))
    X = np.mat(v_ex)
#    X1 = float(X - np.tile(np.mean(X, axis = 0),(46,1))) / np.tile(np.std(X, axis = 0),(46,1))
    
age = mt.loadmat(ag_path)

for k_ag, v_ag in age.items():
    c_ag = k_ag
    y = np.array(v_ag)
    y = y.flatten()

####### the first part 
    
#  obtain the index of feature 
def sele_fea(X,y): # X is the data; y is the age

    
    #X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    estimator = SVR(kernel="linear")
    # step: corresponds to the (integer) number of features to remove at each iteration
    # cv: 为要分成的包的总个数
    # 
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
sel_fea_index_TD = np.load('sel_fea_index_TD_78.npy')

#sel_fea_index_ASD = np.load('sel_fea_index_str_ASD_75.npy')

# reChoose feature after correlation with head motion without regressing site and sex

#sel_fea_index_TD_new = np.delete(sel_fea_index_TD,(22,23,24),axis = 0)
#sel_fea_index_ASD_new = np.delete(sel_fea_index_ASD,0,axis = 0)

# reChoose feature after correlation with head motion with regressing site and sex
#sel_fea_index_TD_new = np.delete(sel_fea_index_TD,(24),axis = 0)
#sel_fea_index_ASD_new = np.delete(sel_fea_index_ASD,(27,30,31),axis = 0)

# intersection
 inter_fea_index = list(set(sel_fea_index_TD).intersection(set(sel_fea_index_ASD)))
 #feature = 5，10 were related with motion (stru & fun)
# inter_fea_index_delMot = np.delete(inter_fea_index,(5,10),axis = 0) 
# feature = 1，2, 5 were related with motion (stru)
  inter_fea_index_delMot = np.delete(inter_fea_index,(5),axis = 0) #5 denote the index of array
 
 
# np.save('inter_fea_index.npy', inter_fea_index)
#union
# union_fea_index = list(set(sel_fea_index_TD).union(set(sel_fea_index_ASD))) # 交集


# start predition for age
#  intersection
 X = X[:,sel_fea_index_TD]
 #union
 X = X[:,inter_fea_index_delMot]
 
def pred_age(X,y):  #def pred_age(X,y): # X is the data; y is the age
    j = 0  
    loo = LeaveOneOut()
    # linear
#    svr = SVR(kernel='linear') 
    # linearSVR
    svr = LinearSVR(tol = 1e-4)
    
    # rbf
    # svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
    # poly
#     svr = SVR(kernel='poly', C=1, degree=1)
    # nubSVR
#     svr = NuSVR(kernel='linear', C=0.1, nu=0.5)
    y_pred = np.zeros([X.shape[0],1])
    for train, test in loo.split(X):
        print(j)
        svr.fit(X[train], y[train]) 
        y_pred[j] = svr.predict(X[test])
        j = j + 1     
    r2_val_ASD = metrics.mean_absolute_error(y[0:59,],  y_pred[0:59,])
    r2_val_HC = metrics.mean_absolute_error(y[59:107,],  y_pred[59:107,])
    return y_pred, r2_val    


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

    
    
    