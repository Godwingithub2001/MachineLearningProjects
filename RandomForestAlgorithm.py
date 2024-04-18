import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

data_value = pd.read_csv('C:\\DATASCIENCE\\MyProjects\\MachineLearningProjects\\dataset\\heart_disease_dataset.csv')
df = pd.DataFrame(data_value)
print(df.to_string())

data_value.info()

print(data_value.isna().sum())

x = data_value.iloc[:,0:10].values
y = data_value.iloc[:,11].values
df2 = pd.DataFrame(x)
print(df2.to_string())

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()

x[:,1] = labelencoder_x.fit_transform(x[:,1])
x[:,2] = labelencoder_x.fit_transform(x[:,2])
x[:,6] = labelencoder_x.fit_transform(x[:,6])
x[:,8] = labelencoder_x.fit_transform(x[:,8])
df3 = pd.DataFrame(x)
print(df3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)



from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion ="entropy")

classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
print("...Prediction..")
df4=pd.DataFrame({"Actual Result-Y":y_test,"Prediction Result":y_pred})
print(df4.to_string())

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy:%.2f'%(accuracy*100))







