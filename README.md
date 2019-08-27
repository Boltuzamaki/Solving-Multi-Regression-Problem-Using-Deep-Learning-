# Solving-Multi-Regression-Problem-Using-Deep-Learning-
If you have large dataset and you don't want to do much feature engineering and just want very few efforts for predicting regression problem . Just follow this repository then your life will be easy :).
# Edit the code where necessary

Now open the Multivariable_Regression.py

I made comment like '#### 1 #####','#### 2 ###### ' .. where you have to make changes and I also mentioned below what changes you have to made.

 - '#### 1 #####'  -- Replace 'Train_price.csv' with the name of your dataset .csv file for train.
 - '#### 2 #####'  -- replace [:,1:11] --> by the number of independent variable from you dataset here [:,1:11] because 
                      my dataset looks like this ![alt text](https://github.com/Boltuzamaki/Solving-Multi-Regression-Problem-Using-Deep-Learning-/blob/master/Images/final.PNG)
-  '#### 3 #####'  -- And also change [:,11:]--> according to your position of dependent variable position
 - '#### 4 #####'  -- Change the build regressor according to your dataset input_dim = number of independent variables and can also change the model to increase accuracy . Like following and can also reduce overfitting by adding Dropouts.
```sh
def build_regressor():   
    regressor = Sequential()
    regressor.add(Dense(units=70))
    regressor.add(Dense(units=35))
    regressor.add(Dense(units=70))
    regressor.add(Dense(units=35))
    regressor.add(Dense(units=70))
    regressor.add(Dense(units=35))
    regressor.add(Dense(units=35))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae','accuracy'])         
    return regressor
``` 

 - '#### 5 #####'   -- Change number of epochs/batch size according to need.
 - '#### 6 #####'   -- Change range according to shape of your dataset if you have let say 10000 training examples and you want the test set have 1000 examples each time for validation then change range as (0,10000,1000)
 - '#### 7 #####'  -- Replace Test_price.csv' with the name of your dataset .csv file for test 
 - '#### 8 #####'  -- Change the shape of X_test_real if it consist any redundant column.
 
# Run the edited file 

