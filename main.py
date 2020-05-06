
import numpy as np
import pandas as pd

import os
import sys

from sklearn.model_selection import KFold,cross_validate
from sklearn import svm

import optuna


trainDf=pd.read_csv("traink.csv")
testDf=pd.read_csv("test.csv")

trainX=trainDf[["Pclass","Age","Sex","Fare","Embarked"]]
trainX=pd.get_dummies(trainX.fillna(trainX.median()))
trainY=trainDf["Survived"]

testX=testDf[["Pclass","Age","Sex","Fare","Embarked"]]
testX=pd.get_dummies(testX.fillna(testX.median()))


def objective(trial):
    C = trial.suggest_uniform('C', 1, 10)
    gamma = trial.suggest_uniform('gamma', 1e-4, 1e-2)
    clf = svm.SVC(C=C, gamma=gamma)
    return -cross_validate(clf,trainX,trainY,cv=6)['test_score'].mean()
study = optuna.create_study(study_name="ex",storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=100)
#study = optuna.load_study(study_name="ex",storage='sqlite:///example.db')
print("\n===\n")
print(study.best_trial)

clf = svm.SVC(C=study.best_params["C"], gamma=study.best_params["gamma"],verbose=True)
clf.fit(trainX,trainY)
testDf["Survived"]=clf.predict(testX)
testDf[["PassengerId","Survived"]].to_csv("./submission.csv",index=False)

