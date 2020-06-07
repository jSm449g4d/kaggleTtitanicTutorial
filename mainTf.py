
import numpy as np
import pandas as pd

import os
import sys

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,Dropout,Conv2D,Conv2DTranspose,\
ReLU,Softmax,Flatten,Reshape,UpSampling2D,Input,Activation,LayerNormalization
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.join("./", __file__)))


#About GPU resources
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for k in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[k], True)

def normDf(df,_min=0,_max=1.0):
    df=(df-_min)/(_max-_min)
    df.loc[df > 1.0,] = 1.0
    df.loc[df < 0.0,] = 0.0
    return df
    

trainDf=pd.read_csv("train.csv")
trainDf["Age"]=normDf(trainDf["Age"],_min=10,_max=80)
trainDf["Fare"]=normDf(trainDf["Fare"],_min=0,_max=500)
trainDf["SibSp"]=normDf(trainDf["SibSp"],_min=0,_max=5)

testDf=pd.read_csv("test.csv")
testDf["Age"]=normDf(testDf["Age"],_min=10,_max=80)
testDf["Fare"]=normDf(testDf["Fare"],_min=0,_max=500)
trainDf["SibSp"]=normDf(trainDf["SibSp"],_min=0,_max=5)

trainX=trainDf[["Pclass","Age","Sex","Fare","Embarked"]]
trainX=pd.get_dummies(trainX.fillna(trainX.median()))
trainY=trainDf["Survived"]
del trainDf

testX=testDf[["Pclass","Age","Sex","Fare","Embarked"]]
testX=pd.get_dummies(testX.fillna(testX.median()))
del testDf

def GEN(input_shape):
    mod=mod_inp = Input(shape=input_shape)
    mod=Dense(64,activation="relu")(mod)
    mod=Dropout(0.25)(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=Dropout(0.25)(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=Dropout(0.25)(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=Dropout(0.25)(mod)
    mod=Dense(64,activation="relu")(mod)
    mod=Dropout(0.25)(mod)
    mod=Dense(1,activation="sigmoid")(mod)
    return keras.models.Model(inputs=mod_inp, outputs=mod)

model=GEN(input_shape=trainX.shape[1:])
model.compile(optimizer='adam',
                loss=keras.losses.binary_crossentropy)
model.summary()
    
model.fit(trainX,trainY,epochs=300)
model.save('myModel.h5')


submissionDf=pd.read_csv("gender_submission.csv")
submissionDf["Survived"]=np.round(model.predict(testX)).astype(np.int64)
submissionDf[["PassengerId","Survived"]].to_csv("submission.csv",index=False)
