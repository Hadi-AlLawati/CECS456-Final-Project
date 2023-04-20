import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from seaborn import distplot
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

#code reference 
#https://medium.com/geekculture/feature-selection-in-machine-learning-correlation-matrix-univariate-testing-rfecv-1186168fac12
# run correlation matrix and plot
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

#histogram , needs work
distplot(df.head())
# show plot
plt.show()

