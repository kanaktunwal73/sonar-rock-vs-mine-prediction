 # IMPORTING THE DEPENDENCIES
import numpy as np # to handle numbers
import pandas as pd  # to read csv file
from sklearn.model_selection import train_test_split # for train and test
from sklearn.linear_model import LogisticRegression # this is a model for prediction
from sklearn.metrics import accuracy_score # to check accuracy of model 

#  DATA COLLECTION AND DATA PROCESSING (loading the dataset to a pandas dataframe)
sonar_data = pd.read_csv("c:/Users/kanak/Downloads/Copy of sonar data.csv",header = None)
print(sonar_data.head()) 
  
# number of rows and columne 
print(sonar_data.shape) 

# statistical measures of our dataset 
print(sonar_data.describe()) 

# how many rocks and mine in our data 
print(sonar_data[60].value_counts()) 

# mean of rock and mine
print(sonar_data.groupby(60).mean()) 

# separating data and labels
x = sonar_data.drop(60,axis=1)
y = sonar_data[60]
print(x)
print(y)

# TRAINING AND TESTING DATA 
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size = 0.1,stratify = y , random_state = 1)
print(x.shape , x_train.shape , x_test.shape) 
print(x_train)
print(y_train)

# MODEL TRAINING ---> LOGISTIC REGRESSION 
model = LogisticRegression() 
print(model.fit(x_train , y_train)) 

# MODEL EVALUTION (accuracy of traing data) 
x_train_prediction = model.predict(x_train)
train_data_accuracy = accuracy_score(x_train_prediction,y_train ) 
print("accuracy on training data :",train_data_accuracy) 

# accuracy on test data 
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test ) 
print("accuracy on testing data :",test_data_accuracy) 

# MAKING A PREDICTIVE SYSTEM 
input_data = (0.0207,0.0535,0.0334,0.0818,0.0740,0.0324,0.0918,0.1070,0.1553,0.1234,0.1796,0.1787,0.1247,0.2577,0.3370,0.3990,0.1647,0.2266,0.3219,0.5356,0.8159,1.0000,0.8701,0.6889,0.6299,0.5738,0.5707,0.5976,0.4301,0.2058,0.1000,0.2247,0.2308,0.3977,0.3317,0.1726,0.1429,0.2168,0.1967,0.2140,0.3674,0.2023,0.0778,0.0925,0.2388,0.3400,0.2594,0.1102,0.0911,0.0462,0.0171,0.0033,0.0050,0.0190,0.0103,0.0121,0.0042,0.0090,0.0070,0.0099)


# changing the input data into numpy array 
input_data_as_numpy_array = np.asarray(input_data) 
input_data_reshaped = input_data_as_numpy_array.reshape(1 , -1) 
prediction = model.predict(input_data_reshaped)
print(prediction) 
if(prediction[0]=='R'):
    print("the object is Rock")
else:
    print("the object is Mine")

