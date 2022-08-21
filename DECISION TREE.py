# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 19:55:01 2022

@author: Rakesh
"""

#################################problem 1################################
###Referred Github coding from Mr Nithin###########################

import pandas as pd
import numpy as np
#loading data set##
company = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/Company_Data.csv')

##dropping age column #
company2 =  company.drop(['Age'], axis=1)


##creating dummy variable for Shelvloc , Urban , US##
company2 = pd.get_dummies(company, columns=['ShelveLoc' , 'Urban' , 'US'])

##converting continous to categorical data#
max = company2['Sales'].max()
company2['Sales']= pd.cut(company2.Sales, bins=[-999, max/2,999], labels=['low', 'high'])

##checking for na or null values#
company2.isna().sum()
company2.isnull().sum()
##no such values#

##split into test and train datasets#
colnames=list(company2.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company2 ,train_size=0.3)

##model building ##
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

##prediction on Test data#
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'],colnames=['Predictions'])
np.mean(preds==test[target]) ##test data accuacy#
#its overfitting model#
##we have to use pruning technique to resolve this overfitting problem#

#dropping age column#
company =  company.drop(['Age'], axis=1)
##creating dummy variable for Shelvloc , Urban , US##
company2 = pd.get_dummies(company, columns=['ShelveLoc' , 'Urban' , 'US'])
##split into test and train datasets#
colnames=list(company2.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company2 ,train_size=0.3)

##train the regression DT#
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth=5 , ccp_alpha=0.05)
regtree.fit(train[predictors], train[target])

#prediction#
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

##measuring accuracy##
from sklearn.metrics import mean_squared_error, r2_score

##error on test data#
mean_squared_error(test[target], test_pred)
r2_score(test[target],test_pred)
##error on train data#
mean_squared_error(train[target], train_pred)
r2_score(train[target],train_pred)

#minimum observation at leaf node approach#
regtree2= tree.DecisionTreeRegressor(min_samples_leaf = 5)
regtree2.fit(train[predictors], train[target])

#prediction#
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

##error on test data#
mean_squared_error(test[target], test_pred2)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred2)

#minimum observation at leaf node approach#
regtree3= tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

#prediction#
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

#error on test data#
mean_squared_error(test[target], test_pred3)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred3)

##random forest technique#

##dropping age column #
company2 =  company.drop(['Age'], axis=1)


##creating dummy variable for Shelvloc , Urban , US##
company2 = pd.get_dummies(company, columns=['ShelveLoc' , 'Urban' , 'US'])

##converting continous to categorical data#
max = company2['Sales'].max()
company2['Sales']= pd.cut(company2.Sales, bins=[-999, max/2,999], labels=['low', 'high'])

##checking for na or null values#
company2.isna().sum()
company2.isnull().sum()

#split into test and train datasets#
colnames=list(company2.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split
train,test = train_test_split(company2 ,train_size=0.3)
##random forest model building ##
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(train[predictors],train[target])

from sklearn.metrics import accuracy_score,confusion_matrix
##accuracy on test data#
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

##accuracy on train data#

confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

##Gridsearch CV#
from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1 , random_state=42)

param_grid = {'max_features':[4,5,6,7,8,9,10], 'min_samples_split': [2,3,10]}

grid_search= GridSearchCV(rf_clf_grid, param_grid,n_jobs=-1,cv=5, scoring='accuracy')

grid_search.fit(train[predictors],train[target])

grid_search.best_params_

cv_rlf_clf_grid = grid_search.best_estimator_
from sklearn.metrics import accuracy_score, confusion_matrix
##testing accuracy#
confusion_matrix(test[target], cv_rlf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rlf_clf_grid.predict(test[predictors]))

##training accuracy#
confusion_matrix(train[target], cv_rlf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rlf_clf_grid.predict(train[predictors]))

##model is now right fit#

##############################################problem 2#################################################

import pandas as pd
import numpy as np

##loading dataset#
diabetes = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/Diabetes.csv')

diabetes.isna().sum()
diabetes.isnull().sum()
##no null values#

##splittig data into test and train set#

colnames=list(diabetes.columns)
predictors=colnames[:8]
target= colnames[8]

from sklearn.model_selection import train_test_split

train,test = train_test_split(diabetes , train_size= 0.3)

##model building #
from sklearn.tree import DecisionTreeClassifier as DT


model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

##test data prediction#
preds= model.predict(test[predictors])
pd.crosstab(test[target], preds,rownames=['Actual'], colnames=['Predictions'] )

np.mean(preds==test[target]) ##test data accuracy#

##train data prediction#
preds= model.predict(train[predictors])
pd.crosstab(train[target], preds,rownames=['Actual'], colnames=['Predictions'] )

np.mean(preds==train[target])

##model is overfitting#

##let us use pruning technique#

##creating dummy variables for categorical data ##
diabetes = pd.get_dummies(diabetes, columns = [" Class variable"])

##spliting data into test and train set#

colnames=list(diabetes.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split

train,test = train_test_split(diabetes , train_size= 0.3)

##train the regression DT#
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth=3 , ccp_alpha=0)
regtree.fit(train[predictors], train[target])

#prediction#
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

##measuring accuracy##
from sklearn.metrics import mean_squared_error, r2_score

##error on test data#
mean_squared_error(test[target], test_pred)
r2_score(test[target],test_pred)
##error on train data#
mean_squared_error(train[target], train_pred)
r2_score(train[target],train_pred)

#minimum observation at leaf node approach#
regtree2= tree.DecisionTreeRegressor(min_samples_leaf = 5)
regtree2.fit(train[predictors], train[target])

#prediction#
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

##error on test data#
mean_squared_error(test[target], test_pred2)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred2)

#minimum observation at leaf node approach#
regtree3= tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

#prediction#
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

#error on test data#
mean_squared_error(test[target], test_pred3)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred3)

##random forest technique#

diabetes = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/Diabetes.csv')

##splitting data into test and train set#

colnames=list(diabetes.columns)
predictors=colnames[:8]
target= colnames[8]

from sklearn.model_selection import train_test_split

train,test = train_test_split(diabetes , train_size= 0.3)

##random forest model building#
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(train[predictors],train[target])

from sklearn.metrics import accuracy_score, confusion_matrix

##accuracy on test data#
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

##accuracy on train data#

confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

##Gridsearch CV#
from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1 , random_state=42)

param_grid = {'max_features':[4,5,6,7,8,9,10], 'min_samples_split': [2,3,10]}

grid_search= GridSearchCV(rf_clf_grid, param_grid,n_jobs=-1,cv=5, scoring='accuracy')

grid_search.fit(train[predictors],train[target])

grid_search.best_params_

cv_rlf_clf_grid = grid_search.best_estimator_
from sklearn.metrics import accuracy_score, confusion_matrix
##testing accuracy#
confusion_matrix(test[target], cv_rlf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rlf_clf_grid.predict(test[predictors]))

##training accuracy#
confusion_matrix(train[target], cv_rlf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rlf_clf_grid.predict(train[predictors]))

####model is now right fit# 
#need rechecking by evaluator and feedback needed##

#################################Problem 3###################################
import pandas as pd
import numpy as np
##loading dataset#
fraud = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/Fraud_check.csv')

##creating dummies for categorical data#
fraud= pd.get_dummies(fraud, columns=['Undergrad','Marital.Status','Urban' ])

##converting taxable income columsn to categorical#
fraud['Taxable.Income'] = pd.cut(fraud['Taxable.Income'], bins=[-999,30000, 99999999], labels=['Risky' ,'Non Risky'])

fraud.isna().sum()
fraud.isnull().sum()
##no any null values#

##splitting data into test and train set#

colnames=list(fraud.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split

train,test = train_test_split(fraud, train_size= 0.3)

##model building##
from sklearn.tree import DecisionTreeClassifier as DT

model = DT(criterion='entropy')
model.fit(train[predictors], train[target])

##test data prediction#
preds= model.predict(test[predictors])
pd.crosstab(test[target], preds,rownames=['Actual'], colnames=['Predictions'] )

np.mean(preds==test[target]) ##test data accuracy#

##train data prediction#
preds= model.predict(train[predictors])
pd.crosstab(train[target], preds,rownames=['Actual'], colnames=['Predictions'] )

np.mean(preds==train[target])
##modell is overfitting##
#using pruning techniques##

##loading dataset#
fraud = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/Fraud_check.csv')

##creating dummies for categorical data#
fraud= pd.get_dummies(fraud, columns=['Undergrad','Marital.Status','Urban' ])

##splitting data into test and train set#

colnames=list(fraud.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split

train,test = train_test_split(fraud, train_size= 0.3)

##train the regression DT#
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth=4)
regtree.fit(train[predictors], train[target])

#prediction#
test_pred = regtree.predict(test[predictors])
train_pred = regtree.predict(train[predictors])

##measuring accuracy##
from sklearn.metrics import mean_squared_error, r2_score

##error on test data#
mean_squared_error(test[target], test_pred)
r2_score(test[target],test_pred)
##error on train data#
mean_squared_error(train[target], train_pred)
r2_score(train[target],train_pred)

#minimum observation at leaf node approach#
regtree2= tree.DecisionTreeRegressor(min_samples_leaf = 5)
regtree2.fit(train[predictors], train[target])

#prediction#
test_pred2 = regtree2.predict(test[predictors])
train_pred2 = regtree2.predict(train[predictors])

##error on test data#
mean_squared_error(test[target], test_pred2)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred2)

#minimum observation at leaf node approach#
regtree3= tree.DecisionTreeRegressor(min_samples_leaf = 2)
regtree3.fit(train[predictors], train[target])

#prediction#
test_pred3 = regtree3.predict(test[predictors])
train_pred3 = regtree3.predict(train[predictors])

#error on test data#
mean_squared_error(test[target], test_pred3)
r2_score(test[target],test_pred2)
##error on train data#
mean_squared_error(train[target], train_pred2)
r2_score(train[target],train_pred3)

#random forest technique#

##converting continous into caetegorical#
fraud['Taxable.Income'] = pd.cut(fraud['Taxable.Income'], bins=[-999,30000, 99999999], labels=['Risky' ,'Non Risky'])

##splitting data into test and train set#

colnames=list(fraud.columns)
predictors=colnames[1:]
target= colnames[0]

from sklearn.model_selection import train_test_split

train,test = train_test_split(fraud, train_size= 0.3)

##random forest building##
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)
rf_clf.fit(train[predictors],train[target])

from sklearn.metrics import accuracy_score, confusion_matrix
##accuracy on test data#
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

##accuracy on train data#
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

##GridSearch CV##
from sklearn.model_selection import GridSearchCV

rf_clf_grid = RandomForestClassifier(n_estimators=500, n_jobs=1 , random_state=42)

param_grid = {'max_features':[4,5,6,7,8,9,10], 'min_samples_split': [2,3,10]}

grid_search= GridSearchCV(rf_clf_grid, param_grid,n_jobs=-1,cv=5, scoring='accuracy')

grid_search.fit(train[predictors],train[target])

grid_search.best_params_

cv_rlf_clf_grid = grid_search.best_estimator_
from sklearn.metrics import accuracy_score, confusion_matrix

##testing accuracy#
confusion_matrix(test[target], cv_rlf_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_rlf_clf_grid.predict(test[predictors]))

##training accuracy#
confusion_matrix(train[target], cv_rlf_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_rlf_clf_grid.predict(train[predictors]))

####model is now right fit# 

#################################problem 4#########################################
import pandas as pd
import numpy as np

###loading dataset##
HR_data = pd.read_csv('D:/DATA SCIENCE ASSIGNMENT/Datasets_DTRF/HR_DT.csv')

##creating dummy values##
HR_data= pd.get_dummies(HR_data, columns= ["Position of the employee"])

##random forst technique#
##splitting data into test and train set#

colnames=list(HR_data.columns)
target=colnames[1]
predictors= colnames[:1]+colnames[2:]

from sklearn.model_selection import train_test_split
train,test = train_test_split(HR_data, train_size= 0.3)

##building random forst building#
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=50)
rf_clf.fit(train[predictors],train[target])

from sklearn.metrics import accuracy_score , confusion_matrix

##accuracy on test data#
confusion_matrix(test[target], rf_clf.predict(test[predictors]))
accuracy_score(test[target], rf_clf.predict(test[predictors]))

##accuracy on train data#
confusion_matrix(train[target], rf_clf.predict(train[predictors]))
accuracy_score(train[target], rf_clf.predict(train[predictors]))

##creating dataframe for employee claim details###

##filliing region manager as 1 as we anyway converted it into dummy here#
employee_claim_list = [[1,5.0,70000]]

employee_claim= pd.DataFrame(employee_claim_list, columns=["Position of the employee" , "no of Years of Experience of employee", " monthly income of employee"])

##concatenating both dataframes#
df = [test,employee_claim]
test=pd.concat(df)

##there is NA in last columns lets fill it with 0#
test= test.fillna(0)

##predicting using test data where employee claim present in last row of test#
preds=rf_clf.predict(test[predictors])

##storing predicted values in separate columsn#
test['predicted salary']= preds

##accesing predicted values
test.iloc[58,13]
# predicted as 112635 , candidate claimed 122391 which is not close.
#recheck is needed as candidate has differed almost 10k as his salary









