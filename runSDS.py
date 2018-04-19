'''
Authors:
Date: 02/2018 
File name: runSDS.py

This program uses the SDS class to perform the Stochastic Diffusion Search alg.
	It creates and setsup the data to begin the alg.


'''

################## IMPORTS ################
from sdsClass import SDS #Performs sds alg.

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import * #the asterisk means everything in it
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.decomposition import PCA #does not look like it is being used
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import *

from imblearn.over_sampling import SMOTE
##############################################


X,y=load_digits(return_X_y=True)

print("X's shape is: " + str(X.shape))

y.shape

np.bincount(y)

#converting dataset to imbalanced dataset
y=np.where(y==1,1,0) 

np.bincount(y)

m_ids=np.where(y==0)

m_ids[0]

X_mclass=X[m_ids[0]] #seperating Majority Class from minoirty class

X_mclass.shape

X_minclass=X[np.where(y==1)[0]]

X_minclass.shape

###### Making Instance of class SDS ########
mySDS = SDS(X_mclass, threshold = 0.5, maxLenOfArrayX = 500, numIterations = 30, numAgents = 100)
#############################################

####### Starting SDS alg. on instance of SDS class ########################
X_SDS, model_ids = mySDS.sdsStart() #sdsStart(X_mclass,threshold=0.5,maxLenOfArrayX=500, numIterations=30, numAgents=100)
########################

X_SDS.shape #299 records retained after applying sds

model_ids[0:5] #models from sds

len(model_ids)

X_models=X_mclass[model_ids]

len(X_models)

X_mclass=np.concatenate([X_SDS,X_models], axis=0) #concatinating remaining records with models to get final majority class

X_mclass.shape #558 records removed after sds

y_mclass=np.zeros(shape=X_mclass.shape[0], dtype=np.int)

y_mclass.shape

y_minclass=np.ones(shape=X_minclass.shape[0], dtype=np.int)

y_minclass.shape

X=np.concatenate([X_mclass,X_minclass], axis=0) #merging minority class with majority class

y=np.concatenate([y_mclass,y_minclass], axis=0)

X.shape

y.shape

np.bincount(y)

y[-239:-1]

#using SMOTE
smote=SMOTE(random_state=0)

#oversampling minority class
X,y=smote.fit_sample(X,y) 

X.shape

y.shape

np.bincount(y) #bincout after undersampling follwed by oversampling

svc=SVC(kernel='linear', C=1) #linear SVM

svc_rbf=SVC(kernel='rbf',gamma=1,C=1) #rbf kernel

X_train,X_test,y_train,y_test=train_test_split(X,y, random_state=0) #splitting the dataset into training and test data

svc.fit(X_train,y_train) #fitting linear kernel to data

svc_rbf.fit(X_train,y_train) #fitting rbf kernel to data

svc.score(X_test,y_test) #accuracy score with linear kernel

svc_rbf.score(X_test,y_test) #accuracy score on test with rbf kernel

svc_rbf.score(X_train,y_train) #classifier is overfitting the training data

train_scores,test_scores=validation_curve(estimator=svc_rbf,X=X,y=y,param_name='gamma', param_range=[0.5,1,2], cv=3) #checking with different gamma values to avoid overfitting

train_scores

test_scores

test_scores.mean(axis=1) #gamma=0.5 gave best test score

gridsearch=GridSearchCV(estimator=svc_rbf,param_grid={'gamma':[0.5,1,2,3], 'C':[0.5,1]},return_train_score=True, cv=3)

gridsearch.fit(X,y) #trying out different combinations of gamma and C to see which gives better values

gridsearch.best_params_ #C=1 and gamma=0.5 are best params

gridsearch.best_score_

#creating new svc with best param combination
svc_rbf1=SVC(kernel='rbf',gamma=0.5, C=1) 

svc_rbf1.fit(X_train,y_train)

#accuracy increased to 0.78 from 0.65
svc_rbf1.score(X_test,y_test) 

y_pred_linear=svc.predict(X_test)

y_pred_rbf=svc_rbf1.predict(X_test)

########### PRINTING REPORT ############
print(classification_report(y_pred_linear,y_test))

print(classification_report(y_pred_rbf,y_test))
######################################

y_linear_thresholds=svc.decision_function(X_test)

y_rbf_thresholds=svc_rbf1.decision_function(X_test)

fpr,tpr, thresholds=roc_curve(y_score=y_linear_thresholds,y_true=y_test)

#linear kernel Aread under curve score
auc(fpr,tpr) 

fpr_rbf,tpr_rbf,thresholds_rbf=roc_curve(y_score=y_rbf_thresholds,y_true=y_test)

#rbf kernel Area under curve
auc(fpr_rbf,tpr_rbf) 

