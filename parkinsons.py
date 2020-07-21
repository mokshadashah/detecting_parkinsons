# code to check accuracy of model to detect parkinsons disease
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read the file
df = pd.read_csv(input("enter file name:"))
print (df.head())

# spitting the data set into features and labes 
features= df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


print("No. of 1's: ", labels[labels==1].shape[0], "\nNo. of 0's: ",labels[labels==0].shape[0])

# transforming the x scale 
scaler = MinMaxScaler((-1,1))
x = scaler.fit_transform(features)
y = labels

# splitting the data into training and test set 
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25, random_state=7)

# using xgbclassifier() we build a model 
#and fit it with out data 
model = XGBClassifier()
model.fit(x_train,y_train)

print (model)

# we get the accuracy score based on the predictions and the test value 
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
print("Accuracy: %.2f%%"  % (accuracy_score(y_test, predictions) *100))
