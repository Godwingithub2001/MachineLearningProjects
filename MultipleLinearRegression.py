import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp

data = pd.read_csv("C:\\DATASCIENCE\MyProjects\\MachineLearningProjects\\dataset\\life_expectancy.csv")
print(data.to_string())

data.info()

print(data.isna().sum())

data = data.drop(['Population'], axis=1)

x=data.iloc[:,0:6].values
print(x)

y = data.iloc[:,6]
print(y)

df1 = pd.DataFrame(x)
df2 = pd.DataFrame(y)
print("List")
print(df1.to_string())

print("life Expectancy")
print(df2.to_string())

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()

x[:,1] = labelencoder_x.fit_transform(x[:,1])
x[:,2] = labelencoder_x.fit_transform(x[:,2])
df3 = pd.DataFrame(x)
print(df3)




# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(df.to_string())

print("Mean")
print(data.describe())

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")
k_test=[[1.0,0.0,142107.34,91391.77,366168.42]]


