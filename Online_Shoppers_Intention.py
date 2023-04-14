import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv('online_shoppers_intention.csv')
print(df.head())

numerical_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
categorical_features = list(df.select_dtypes(include=['object']).columns)
print(numerical_features)
print(categorical_features)

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