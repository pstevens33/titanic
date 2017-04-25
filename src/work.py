from gradient_descent import GradientDescent
import logistic_regression_functions as f
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import LogisticRegression, ElasticNetCV, SGDClassifier


################# Read in Data and Remove unnecessary columns and missing rows (Train) ###############
X = pd.read_csv('../data/train.csv')

X.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
X['family'] = X['SibSp'] + X['Parch']
X.drop(['SibSp', 'Parch'], axis=1, inplace=True)
null_data = X[X.isnull().any(axis=1)]
null_indexes = null_data.index.values
X_temp = X[X['Age'].notnull()]
# print(X_temp.Age.mean())
X.loc[:, "Age"] = X.loc[:, "Age"].fillna(X_temp.Age.mean())
#X.drop(null_indexes, axis=0, inplace=True)
X = pd.get_dummies(X)
X.drop(['Sex_male'], axis=1, inplace=True)
#X['Age'] = np.log1p(X['Age'])
X = X[X['Age'] > 2.6]
y = X['Survived']
#print(y.sum()/len(y))


################# Read in Data and Remove unnecessary columns and missing rows (Test) ###############
X_test = pd.read_csv('../data/test.csv')

Ids = X_test['PassengerId']
X_test.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)
X_test['family'] = X_test['SibSp'] + X_test['Parch']
X_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)
null_data = X_test[X_test.isnull().any(axis=1)]
null_indexes = null_data.index.values
X_temp = X_test[X_test['Age'].notnull()]
X_temp_fare = X_test[X_test['Fare'].notnull()]
X_test.loc[:, "Age"] = X_test.loc[:, "Age"].fillna(X_temp.Age.mean())
X_test.loc[:, "Fare"] = X_test.loc[:, "Fare"].fillna(X_temp.Fare.mean())
#X.drop(null_indexes, axis=0, inplace=True)
X_test = pd.get_dummies(X_test)

X_test.drop(['Sex_male'], axis=1, inplace=True)

#X_test['Age'] = np.log1p(X_test['Age'])


################## Plot histograms of features ###########################
# fig = plt.figure(figsize=(8,6))
# for count in range(6):
#     ax = fig.add_subplot(2,3,count+1)
#     ax.hist(X.iloc[:,count])
#     ax.set_title('{}'.format(X.columns[count]))
# fig.tight_layout()
# fig.savefig('../plots/master_hist.png')
#
#
# sns.set()
# sns.pairplot(X, hue="Survived")
# plt.savefig('../plots/scatter_matrix.png')



############## Look at the correlations to get an estimate of feature importance #############
corr = X.corr()
#corr.sort_values(['Fare'], ascending = False, inplace=True)
# print(corr.Fare)
X.drop('Survived', axis=1, inplace=True)

############################ RMSE Scoring Function ###############################

scorer = make_scorer(mean_squared_error, greater_is_better=False)

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring=scorer, cv=20))
    return(rmse)


############################ Base Logistic Model ###############################
# logistic_model = LogisticRegression(penalty='l1')
# logistic_model.fit(X,y)
# y_logistic = logistic_model.predict(X)
# #print(np.mean(rmse_cv(logistic_model)))
# print("# of correct guesses out of 822: {}".format(np.sum(y==y_logistic)))
# #print(logistic_model.fit(X,y).coef_)

############################ Elastic Net Logistic Model ###############################
elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006,
                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 50000, cv = 10)
elasticNet.fit(X, y)
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
# print("Best l1_ratio :", ratio)
# print("Best alpha :", alpha )
#
# print("Try again for more precision with l1_ratio centered around " + str(ratio))
elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6],
                          max_iter = 50000, cv = 10)
elasticNet.fit(X, y)
if (elasticNet.l1_ratio_ > 1):
    elasticNet.l1_ratio_ = 1
alpha = elasticNet.alpha_
ratio = elasticNet.l1_ratio_
# print("Best l1_ratio :", ratio)
# print("Best alpha :", alpha )
#
# print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) +
#       " and alpha centered around " + str(alpha))
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
# print("Best l1_ratio :", ratio)
# print("Best alpha :", alpha )

#print("RMSE: {}".format(rmse_cv(elasticNet).mean()))

for i in np.arange(0,1,0.001):
    result_train = np.ceil(elasticNet.predict(X)-i).astype(int)
    print("ElasticNet {} # of correct guesses out of 822: {}".format(i, np.sum(y==result_train)))
result_test = np.ceil(elasticNet.predict(X_test)-0.62).astype(int)
result_df = pd.DataFrame(result_test)


############################# Gradient Descent ##############################
#
# gd = GradientDescent(f.cost_regularized, f.gradient_regularized, f.predict)
# gd.fit_stochastic(X, y)
# ypred = gd.predict(X)
# print("Regularized, Stochastic Gradient Descent: {}".format(np.sum(y==ypred)))
#
# gd = GradientDescent(f.cost, f.gradient, f.predict, fit_intercept=False)
# gd.fit(X, y)
# ypred = gd.predict(X)
# print("No Intercept Gradient Descent: {}".format(np.sum(y==ypred)))
#
# gd = GradientDescent(f.cost_regularized, f.gradient_regularized, f.predict, fit_intercept=False)
# gd.fit(X, y)
# ypred = gd.predict(X)
# print("No Intercept, Regularized Gradient Descent: {}".format(np.sum(y==ypred)))
#
# gd = GradientDescent(f.cost_regularized, f.gradient_regularized, f.predict, fit_intercept=False)
# gd.fit_stochastic(X, y)
# ypred = gd.predict(X)
# print("No Intercept, Regularized, Stochastic Gradient Descent: {}".format(np.sum(y==ypred)))
# result_df = pd.DataFrame(ypred)

################################# Submission ############################
Ids = pd.DataFrame(Ids)
final = pd.concat([Ids, result_df], axis=1)
final.columns = ['PassengerId', 'Survived']
final.set_index('PassengerId', inplace=True)
final.to_csv('../data/submission3.csv')
