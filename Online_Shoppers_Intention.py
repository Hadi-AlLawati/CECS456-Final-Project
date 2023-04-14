import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('online_shoppers_intention.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
numerical_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                      'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues',
                      'SpecialDay', 'Weekend', 'Revenue']


df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)


#handle missing values
imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])
imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = imputer.fit_transform(df[categorical_features])

#use onehotencoder to change categorical features to numerical, like false to 0 and true to 1
onehotencoder = OneHotEncoder(handle_unknown='ignore')
df = pd.get_dummies(df, columns=categorical_features)

#scaling the numerical values so it is not stupid
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print(df.head())