import pandas as pd

train = pd.read_csv('../data/train.csv')
train.drop('Name', axis=1, inplace=True)
missing = train.isnull().sum()
missing = missing[missing > 0]
print(missing)
