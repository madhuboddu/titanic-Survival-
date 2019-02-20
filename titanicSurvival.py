#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 15:36:55 2019

@author: madhusudhanreddy
"""

#data analysis and warangling
import pandas as pd
import numpy as np
import random as rnd

#data visualization
import seaborn as sns
import matplotlib.pyplot as plt

#machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC , LinearSVC
from sklearn.ensemble import RandomForestClassifier


train_df =pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

combine = [train_df, test_df]

print(train_df.columns.values)

train_df.head()

train_df.info()

#Categorical: Survived, Sex, and Embarked. 
#Ordinal: Pclas

#Continous: Age, Fare. 
#Discrete: SibSp, Parch.

#Check for various data trends

# 1 Which features contain blank, null or empty values?
    #check for all the numm values.
# 2 Find for mixed type of data 
    #Cabin has alpha numberic data.
# 3 Find errors or typos
    #name may contain error or typos.
# 4 Find various type of data types of each datas.

train_df.info()



train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex']].groupby(['Sex'])


#various plots to understand the data.
sns.factorplot('Pclass',data=train_df,kind='count',hue='Sex' )

sns.factorplot('Sex',data=train_df,kind='count')

sns.factorplot('Cabin',data=train_df,kind='count')

sns.factorplot('Survived',data=train_df,kind='count',hue='Sex')

#indexes of people who did not survive.
not_survived = train_df[train_df['Survived']== 0]

#no of people whi didnot survive.
len(not_survived)

not_survived_pvt = not_survived.pivot_table('Survived','Sex','Pclass', margins= True,aggfunc = len)

table = pd.crosstab(index=[train_df.Survived,train_df.Pclass] , columns = [train_df.Sex,train_df.Embarked])

table.unstack()







