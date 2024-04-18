import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('C:\\DATASCIENCE\\MyProjects\\MachineLearningProjects\\dataset\\indian_food.csv')
dataset.dropna(inplace=True)
print(dataset.isna().sum())

df = pd.DataFrame(dataset)
print(df.to_string())

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
dataset[['name', 'diet','state']] = dataset[['name', 'diet','state']].apply(LabelEncoder().fit_transform)
print(dataset)

x = dataset.iloc[:,[0,2]].values
df2 = pd.DataFrame(x)
print(df2.to_string())

y = dataset.iloc[:,7].values
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier

classifier= KNeighborsClassifier(n_neighbors=5, metric='minkowski',
p=2 )
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)
print(y_pred)

print("Prediction Comparison")
ddf = pd.DataFrame({"Y_test":y_test, "Y_pred":y_pred})
print(ddf.to_string())

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' %(accuracy*100))

test=[[226,1]]
pred = classifier.predict(test)
print(pred)











