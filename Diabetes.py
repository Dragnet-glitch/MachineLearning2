#import dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
#loading the data set
data = pd.read_csv('diabetes.csv')
#no of rows and columns
#print(data.shape)
#print(data.describe())
#print(data['Age'].value_counts())
#0 reps non diabetic
#1 reps diabetic
#print(data.groupby('Outcome').mean())
#seperating data and labels
x = data.drop(columns= 'Outcome', axis = 1)
y = data['Outcome']
#print(x)
#print(y)
#data standardization
scaler = StandardScaler()
scaler.fit(x)

standardized_data = scaler.transform(x)

x = standardized_data
y = data['Outcome']




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

#print(x_test)

#print(x.shape, x_train.shape, x_test.shape)

#Training the moel
classifier = svm.SVC(kernel='linear')
#training the support veector Machine Classifier
classifier.fit(x_train, y_train)

#model evaluation
x_train_prediction = classifier.predict(x_train)
accuracy = accuracy_score(x_train_prediction, y_train)
print(accuracy)

x_test_prediction = classifier.predict(x_test)
accuracy = accuracy_score(x_test_prediction, y_test)
print(accuracy)
#building the predictive system
input_data = (4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
#print(prediction)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")