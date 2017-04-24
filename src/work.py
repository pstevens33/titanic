import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LogisticRegression, ElasticNetCV


################# Read in Data and Remove unnecessary columns and missing rows ###############
X = pd.read_csv('../data/train.csv')
X.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
null_data = X[X.isnull().any(axis=1)]
null_indexes = null_data.index.values
X.drop(null_indexes, axis=0, inplace=True)
X = pd.get_dummies(X)
y = X['Survived']
print(X.head())

############## Look at the correlations to get an estimate of feature importance #############
corr = X.corr()
corr.sort_values(['Survived'], ascending = False, inplace=True)
print(corr.Survived)
X.drop('Survived', axis=1, inplace=True)

############################ RMSE Scoring Function ###############################

scorer = make_scorer(mean_squared_error, greater_is_better=False)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring=scorer, cv=10))
    return(rmse)

############################ Base Logistic Model ###############################
logistic_model = LogisticRegression()
print(np.mean(rmse_cv(logistic_model)))
#print(logistic_model.fit(X,y).coef_)

############################ Elastic Net Logistic Model ###############################
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 50000, cv = 10)
elasticNet.fit(X, y)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 50000, cv = 10)
elasticNet.fit(X, y)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
      " and alpha centered around " + str(alpha))
elasticNet = ElasticNetCV(l1_ratio = ratio,
                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9,
                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3,
                                    alpha * 1.35, alpha * 1.4],
                          max_iter = 50000, cv = 10)
elasticNet.fit(X, y)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
print("Best l1_ratio :", ratio)
print("Best alpha :", alpha )

print(rmse_cv(elasticNet).mean())
