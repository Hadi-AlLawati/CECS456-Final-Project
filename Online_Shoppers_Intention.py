import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




df = pd.read_csv('online_shoppers_intention.csv')
print(df.head())

numerical_features = list(df.select_dtypes(include=['int64', 'float64']).columns)
print(numerical_features)

df[numerical_features].hist(bins=50, figsize=(20,15))
plt.show()