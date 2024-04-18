
import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
from sklearn import metrics

data_set = pd.read_csv("C:\\DATASCIENCE\\dataset\\csv\\Eligibility_loan_prediction.csv")
print(data_set.describe())

print("Dataset")
df = pd.DataFrame(data_set)
print(df.to_string())

data_set.info()

data_set = data_set.drop(['Loan_ID', 'Gender', 'Married'], axis=1)

print(data_set.isna().sum())

print(data_set.dropna())
print(data_set.isnull().sum())

data_set.info()

d = data_set.replace('3+',0)
print(d)

feature_cols = ['Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term',
                'Credit_History', 'Property_Area']


data_set = data_set.dropna(axis=0)
print(data_set.isnull().sum())



data_set['Education'].replace(['Graduate', 'Not Graduate'],
                        [0, 1], inplace=True)
data_set['Property_Area'].replace(['Rural', 'Urban', 'Semiurban'],
                        [0, 1, 2], inplace=True)
data_set['Self_Employed'].replace(['No', 'Yes'],
                        [0, 1], inplace=True)
data_set['Dependents'].replace([0, 1, 2,'3+'],
                        [0, 1, 2,3], inplace=True)
data_set['Loan_Status'].replace(['Y','N'],
                        [0, 1], inplace=True)

x = data_set[feature_cols]
y = data_set.Loan_Status
df1 = pd.DataFrame(x)
print("X Data")
print(df1.to_string())
df2 = pd.DataFrame(y)
print("Y Data")
print(df2.to_string())

df1 = pd.DataFrame(x)
print("X Data")
print(df1.to_string())
df2 = pd.DataFrame(y)
print("Y Data")
print(df2.to_string())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(x,y,test_size=0.25,random_state=0)
print(X_train)
from sklearn.linear_model import LogisticRegression
#
logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
#
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
df2 = pd.DataFrame(X_test)
#
print(df2.to_string())

print(y_pred)

df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df2.to_string())


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))







