import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error

# Importing the dataset
dataset = pd.read_csv('Train_price.csv')                                                      #########    1   ##########
X = dataset.iloc[:, 1:11].values          # Separating dependent and independent variable     #########    2   ##########
y = dataset.iloc[:, 11:].values                                                               #########    3   ##########          

# Scaling dataset between 0 - 1  
preprocess = MinMaxScaler()               # Scaling for fast converging
preprocess.fit(X)
X_pre = preprocess.transform(X)
preprocess1 = MinMaxScaler()
preprocess1.fit(y)
y_pre = preprocess1.transform(y)


# creating the model
def build_regressor():                                                                        #########    4    ###########                      
    regressor = Sequential()
    regressor.add(Dense(units=12, input_dim=10))
    regressor.add(Dense(units=7))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])         
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
regressor = KerasRegressor(build_fn=build_regressor, batch_size=16,epochs=100)               ##########   5    #############

RMSE =[]               # making list for RMSE to store RMSE of different training sets after ecaluation on test
# looping over ten different sets of training created from a same training set 
for i in range(0,1000,100):                                                                  ##########    6   #############
     X_train = X_pre[0:i]
     X_train1 = ( X_pre[i+100:] )
     X_train_final = np.concatenate((X_train, X_train1))
     y_train = y_pre[0:i]
     y_train1 = ( y_pre[i+100:] )
     y_train_final = np.concatenate((y_train, y_train1))
     X_test_final = X_pre[i:i+100]
     y_test_final = y_pre[i:i+100]
     
     
     results=regressor.fit(X_train_final,y_train_final)
     y_pred= regressor.predict(X_test_final)
     y_pred1 = y_pred.reshape(-1,1)
     
   
     RMSE1 = mean_squared_error(y_test_final, y_pred1)**0.5
     RMSE.append(RMSE1)
     
     

# predicting the real test dataset ------------------------------

# Importing the dataset for testing 
dataset = pd.read_csv('Test_price.csv')                                                       ########     7    #############                      
X_test_real = dataset.iloc[:, 1:].values                                                      ########     8    #############                  
preprocess_test = MinMaxScaler()
preprocess_test.fit(X_test_real)

X_pre_real = preprocess_test.transform(X_test_real) 
y_pred_real= regressor.predict(X_pre_real)

y_pred_real1 = y_pred_real.reshape(-1,1)
y_test_real_final = preprocess1.inverse_transform(y_pred_real1) # making real predicted data from scaled data 

# calculating average RMSE of model
np.average(RMSE1)
