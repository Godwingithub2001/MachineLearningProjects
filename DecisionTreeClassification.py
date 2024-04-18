import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

flip_df = pd.read_csv("C:/DATASCIENCE/MyProjects/MachineLearningProjects/dataset/Flipkart.csv")
print(flip_df.to_string)

flip_df.info()

print(flip_df.describe())

flip_df.dropna(inplace=True)

print(flip_df.isnull().sum())

# Drop the star_5f, star_4f, star_3f, star_2f, and star_1f columns
# Drop the norating1 and noreviews1 columns

flip_df = flip_df.drop(['star_5f','star_4f','star_3f','star_2f','star_1f','norating1','noreviews1'], axis=1)

print(flip_df.info())

# Convert the platform column to numeric
flip_df['platform'] = flip_df['platform'].map({'Flipkart': 0, 'Amazon': 1})

print(flip_df['platform'].unique())

print(flip_df.to_string)

x = flip_df.iloc[:,[0,1,2,3,5,6]].values
y = flip_df.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()

x[:,1] = labelencoder_x.fit_transform(x[:,1])
x[:,3] = labelencoder_x.fit_transform(x[:,3])
flip = pd.DataFrame(x)
print(flip)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)

#Fitting Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)

#Predicting the test set result
y_pred = classifier.predict(x_test)
df2 = pd.DataFrame({"Actual Y_Test":y_test,"Prediction_Data":y_pred})
print("Prediction Result")
print(df2.to_string())

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('Accuracy: %.2f' %(accuracy*100))











