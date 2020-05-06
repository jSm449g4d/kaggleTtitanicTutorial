
import numpy as np
import pandas as pd

import os
import sys

from sklearn.model_selection import KFold,cross_validate
from sklearn import svm

import optuna



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))

def outlierDf(df):
    #q_low = df.quantile(0.05)
    #q_hi  = df.quantile(0.95)
    #df.median()
    tDfn=(df-df.min())/(df.max()-df.min())
    return tDfn

trainDf=pd.read_csv("train.csv")[["Pclass","Age","Sex","Fare","Embarked","Survived"]].dropna()
trainDf[["Age","Fare"]]=outlierDf(trainDf[["Age","Fare"]])
testDf=pd.read_csv("test.csv")
testDf[["Age","Fare"]]=outlierDf(testDf[["Age","Fare"]])

trainX=trainDf[["Pclass","Age","Sex","Fare","Embarked"]]
#trainX=pd.get_dummies(trainX.fillna(trainX.median()))
trainX=pd.get_dummies(trainX)
trainY=trainDf["Survived"]

testX=testDf[["Pclass","Age","Sex","Fare","Embarked"]]
testX=pd.get_dummies(testX.fillna(testX.median()))


def objective(trial):
    C = trial.suggest_uniform('C', 1, 10)
    gamma = trial.suggest_uniform('gamma', 1e-4, 1e-2)
    clf = svm.SVC(C=C, gamma=gamma)
    return -cross_validate(clf,trainX,trainY,cv=6)['test_score'].mean()
study = optuna.create_study(study_name="ex",storage='sqlite:///example.db', load_if_exists=True)
study.optimize(objective, n_trials=100, n_jobs=4)
#study = optuna.load_study(study_name="ex",storage='sqlite:///example.db')
print("\n===\n")
print(study.best_trial)

clf = svm.SVC(C=study.best_params["C"], gamma=study.best_params["gamma"],verbose=True)
clf.fit(trainX,trainY)
testDf["Survived"]=clf.predict(testX)
testDf[["PassengerId","Survived"]].to_csv("./submission.csv",index=False)

