
import numpy as np
import pandas as pd

import os
import sys

from sklearn.model_selection import KFold,cross_val_score
from sklearn import svm


trainDf=pd.read_csv("titanic/train.csv")
testDf=pd.read_csv("titanic/test.csv")

trainX=trainDf[["Pclass","Age","Fare"]]#
trainX=trainX.fillna(trainX.median())
trainY=trainDf["Survived"]

testX=testDf[["Pclass","Age","Fare"]]#
testX=testX.fillna(testX.median())

clf = svm.SVC( kernel='rbf',C=100, gamma=0.001)
clf.fit(trainX,trainY)
# result=cross_val_score(clf,trainX,trainY,cv=6)

testDf["Survived"]=clf.predict(testX)

testDf[["PassengerId","Survived"]].to_csv("./submission.csv",index=False)
