
import numpy as np
import pandas as pd

import os
import sys

from sklearn.model_selection import KFold,cross_validate
from sklearn import svm

import optuna


trainDf=pd.read_csv("train.csv")
testDf=pd.read_csv("test.csv")

trainX=trainDf[["Pclass","Age","Fare"]]
trainX=trainX.fillna(trainX.median())
trainY=trainDf["Survived"]

testX=testDf[["Pclass","Age","Fare"]]
testX=testX.fillna(testX.median())

# clf = svm.SVC( kernel='rbf',C=100, gamma=0.001)
# clf.fit(trainX,trainY)
# result=cross_val_score(clf,trainX,trainY,cv=6)
def objective(trial):
    # C
    C = trial.suggest_uniform('C', 10, 1000)
    gamma = trial.suggest_uniform('gamma', 1e-4, 1e-2)
    clf = svm.SVC(kernel='rbf',C=C, gamma=gamma)
    return -cross_validate(clf,trainX,trainY,cv=6)['test_score'].mean()

#study = optuna.create_study(study_name="ex",storage='sqlite:///example.db', load_if_exists=True)
#study.optimize(objective, n_trials=100)
study = optuna.load_study(study_name="ex",storage='sqlite:///example.db')
print("\n===\n")
print(study.best_trial)


clf = svm.SVC( kernel='rbf',C=study.best_params["C"], gamma=study.best_params["gamma"])
clf.fit(trainX,trainY)
testDf["Survived"]=clf.predict(testX)
testDf[["PassengerId","Survived"]].to_csv("./submission.csv",index=False)

#svm.SVC(kernel='rbf',C=study.C, gamma=gamma)
#testDf["Survived"]=clf.predict(testX)


#clf = svm.SVC( kernel='rbf',C=100, gamma=0.001)
#clf.fit(trainX,trainY)
#result=cross_validate(clf,trainX,trainY,cv=6)['test_score'].mean()

#print('params:', study.best_params)
#print(result)


#testDf["Survived"]=clf.predict(testX)

#testDf[["PassengerId","Survived"]].to_csv("./submission.csv",index=False)
