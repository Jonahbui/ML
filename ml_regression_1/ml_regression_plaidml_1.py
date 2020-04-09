# Author: Jonah Bui
# Date: 4/08/2020
# Description: Used to create and train a linear regression model to predict student grades
# Changelog: --
import plaidml
from plaidml import keras
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD, adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import Model

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

prediction_filename = 'student-mat.csv'
test_filename = 'student-por.csv'
#--------------------------------------------------------------------------------------------------
# Input
#--------------------------------------------------------------------------------------------------
# Import train data and test data
train_data = pd.read_csv(prediction_filename, sep=';')
test_data = pd.read_csv(test_filename, sep=';')
# One hot encode 
train_data.replace(to_replace='no', value=0, inplace=True)
train_data.replace(to_replace='yes', value=1, inplace=True)

test_data.replace(to_replace='no', value=0, inplace=True)
test_data.replace(to_replace='yes', value=1, inplace=True)
# 1. School
train_data.replace(to_replace='GP', value=0, inplace=True)
train_data.replace(to_replace='MS', value=1, inplace=True)

test_data.replace(to_replace='GP', value=0, inplace=True)
test_data.replace(to_replace='MS', value=1, inplace=True)

# 2. Sex
train_data.replace(to_replace='M', value=0, inplace=True)
train_data.replace(to_replace='F', value=1, inplace=True)

test_data.replace(to_replace='M', value=0, inplace=True)
test_data.replace(to_replace='F', value=1, inplace=True)

# 3. Age

# 4. Address
train_data.replace(to_replace='U', value=0, inplace=True)
train_data.replace(to_replace='R', value=1, inplace=True)

test_data.replace(to_replace='U', value=0, inplace=True)
test_data.replace(to_replace='R', value=1, inplace=True)

# 5. Family Size
train_data.replace(to_replace='LE3', value=0, inplace=True)
train_data.replace(to_replace='GT3', value=1, inplace=True)

test_data.replace(to_replace='LE3', value=0, inplace=True)
test_data.replace(to_replace='GT3', value=1, inplace=True)

# 6. Parental Status
train_data.replace(to_replace='T', value=0, inplace=True)
train_data.replace(to_replace='A', value=1, inplace=True)

test_data.replace(to_replace='T', value=0, inplace=True)
test_data.replace(to_replace='A', value=1, inplace=True)

# 7/8. Mother's and Father Education

# 9/10. Mother's and Father's Jobs
train_data.replace(to_replace='other', value=0, inplace=True)
train_data.replace(to_replace='at_home', value=1, inplace=True)
train_data.replace(to_replace='health', value=2, inplace=True)
train_data.replace(to_replace='services', value=3, inplace=True)
train_data.replace(to_replace='teacher', value=4, inplace=True)

test_data.replace(to_replace='other', value=0, inplace=True)
test_data.replace(to_replace='at_home', value=1, inplace=True)
test_data.replace(to_replace='health', value=2, inplace=True)
test_data.replace(to_replace='services', value=3, inplace=True)
test_data.replace(to_replace='teacher', value=4, inplace=True)

# 11. Reason
train_data.replace(to_replace='home', value=0, inplace=True)
train_data.replace(to_replace='reputation', value=1, inplace=True)
train_data.replace(to_replace='course', value=2, inplace=True)
train_data.replace(to_replace='other', value=3, inplace=True)

test_data.replace(to_replace='home', value=0, inplace=True)
test_data.replace(to_replace='reputation', value=1, inplace=True)
test_data.replace(to_replace='course', value=2, inplace=True)
test_data.replace(to_replace='other', value=3, inplace=True)

# 12. Guardian
train_data.replace(to_replace='mother', value=0, inplace=True)
train_data.replace(to_replace='father', value=1, inplace=True)
train_data.replace(to_replace='other', value=2, inplace=True)

test_data.replace(to_replace='mother', value=0, inplace=True)
test_data.replace(to_replace='father', value=1, inplace=True)
test_data.replace(to_replace='other', value=2, inplace=True)

# 13. Travel Time
# 14. Study Time
# 15. Failures

# 16. School Support --
# 17. Family Support --
# 18. Paid --
# 19. Activities --
# 20. Nursery --
# 21. Higher --
# 22. Internet --
# 23. Romantic
# 24. Family Relationship
# 25. Free Time
# 26. Go Out
# 27. Weekday Alcohol
# 28. Weekend Alcohol
# 29. Health
# 30. Absences
print(train_data[:10])
variables = [
    #'school',#1
    #'sex',#2
    'age',#3
    #'address',#4
    #'famsize',#5
    #'Pstatus',#6
    'Medu',#7
    'Fedu',#8
    #'Mjob',#9
    #'Fjob',#10
    #'reason',#11
    #'guardian',#12
    'traveltime',#13
    'studytime',#14
    'failures',#15
    #'schoolsup',#16
    #'famsup',#17
    #'paid',#18
    'activities',#19
    #'nursery',#20
    'higher',#21
    'internet',#22
    #'romantic',#23
    'famrel',#24
    'freetime',#25
    #'goout',#26
    #'Dalc',#27
    #'Walc',#28
    'health',#29
    'absences',#30
    'G1',#31
    'G2',#32
    'G3' #33

]
train_data = train_data[variables]
test_data = test_data[variables]
# Split train data
x_train = np.array(train_data.drop(['G3'], 1))
y_train = np.array(train_data['G3']) 
input_dim = len(x_train[0])

# Split test data
x_test = np.array(test_data.drop(['G3'], 1))
y_test = np.array(test_data['G3'])

# Normalize data
x_train = x_train.astype(float)
y_train = y_train.astype(float)
x_test = x_train.astype(float)
y_test = y_train.astype(float)

earlystop = EarlyStopping(
    monitor = 'val_mean_squared_error',
    patience=20,
    verbose=1
)

modelcheckpoint = ModelCheckpoint(
    filepath='model_regression_plaidml_1.hdf5',
    monitor='val_mean_squared_error',
    verbose=0,
    save_best_only=True 
)# Save the file after improvement of an epoch 

#--------------------------------------------------------------------------------------------------
# Model
#--------------------------------------------------------------------------------------------------
print(f'The shape is {x_train[0].shape}')
model = Sequential([
    Dense(2048, input_dim=len(x_train[0]), kernel_initializer='normal', activation='relu'),
    Dense(8, input_dim=len(x_train[0]), kernel_initializer='normal', activation='relu'),
    Dense(1, kernel_initializer='normal', activation='linear')
])
#--------------------------------------------------------------------------------------------------
# Compile Model
#--------------------------------------------------------------------------------------------------
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

#--------------------------------------------------------------------------------------------------
# Train Model
#--------------------------------------------------------------------------------------------------
history = model.fit(
    x_train, 
    y_train, 
    epochs=2000, 
    batch_size=64, 
    verbose=1, 
    validation_split=0.2, 
    callbacks=[earlystop, modelcheckpoint]
)
plt.plot(history.history['loss'], label='training')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()
#--------------------------------------------------------------------------------------------------
# Test Model
#--------------------------------------------------------------------------------------------------
test_history = model.evaluate(x_test, y_test, batch_size=32)
print(test_history)

#mylist = pred[0]
#i = 0
#for prediction in pred:
#    print(f"Predicted = {prediction} Actual = {y_test[i]}")
#    i+=1

#--------------------------------------------------------------------------------------------------
# Predict 
#--------------------------------------------------------------------------------------------------
test = np.array([15, 4, 4, 1, 4, 0, 1, 1, 1, 5, 1, 5, 0, 20, 20])
test = test.reshape(1,15)
pred = model.predict(test)
print(pred)