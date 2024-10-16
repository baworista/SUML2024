import numpy as np
import pandas as pd
import sklearn.metrics as metrics

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print(train_df.head())
print(train_df.info())

X_train = train_df.iloc[:, 1:-2].values
y_train=train_df.iloc[:, -1].values

print(X_train)
print(y_train)

