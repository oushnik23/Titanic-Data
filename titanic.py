# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 20:24:58 2021

@author: Administrator
"""

import os
os.getcwd()
os.chdir("C:/Users/Administrator/Desktop/PYTHON\Titanic")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

t_train=pd.read_csv("train.csv")
t_test=pd.read_csv("test.csv")
t_train.columns
t_test.columns

t_train.isnull().any()
t_train.info()

t_train['Age']=t_train['Age'].fillna(t_train.Age.mean())
t_train['Embarked']=t_train['Embarked'].fillna('S')
t_train['Cabin']=t_train['Cabin'].fillna(t_train.Cabin.mode()[1])

x=t_train.drop(['Survived','Cabin','Name','Ticket'],axis=1)
y=t_train[['Survived']]

x=pd.get_dummies(x)

t_train['Survived'].value_counts()
t_train['Sex'].value_counts()

#men=t_train.loc[t_train.Sex=='Male']['Survived']
#m=sum(men)/len(men)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.2)

from sklearn.linear_model import LogisticRegression
lr_model=LogisticRegression()
lr_model.fit(x,y)

y_pred=lr_model.predict(x)

from sklearn.metrics import classification_report
print(classification_report(y,y_pred))

t_test.columns
t_test['Parch']

#test data
t_test=t_test.drop(['Cabin','Name','Ticket'],axis=1)
t_test.isnull().sum()
t_test['Age']=t_test['Age'].fillna(t_test.Age.mean())
t_test['Fare']=t_test['Fare'].fillna(t_test.Fare.mean())

test_data=pd.get_dummies(t_test)
prediction=lr_model.predict(test_data)

output = pd.DataFrame({'PassengerId': t_test.PassengerId, 'Survived': prediction})
output.to_csv('my_submission.csv', index=False)
