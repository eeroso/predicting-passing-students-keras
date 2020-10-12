import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
np.set_printoptions(precision=3, suppress=True)
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD


csv_student = 'Student.csv'
datasets = pd.read_csv(csv_student, delimiter=',', header=1) #read the file, first row for header

dataset = datasets.values 

X = dataset[:,0:2] #slice input data

Y = dataset[:,3] #output data

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X) #map between 0 and 1





X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3) #Training data 70%, testing data 30%
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

#2 hidden layers, 18 neurons in each
model = Sequential([Dense(18, activation='relu', input_shape=(2,)), Dense(18, activation='relu'), Dense(1, activation='sigmoid'),]) #sigmoid activation for last layer to get between 0 and 1
opt = tf.keras.optimizers.Adam(learning_rate=0.003) #adam optimizer with tweaked learning rate
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(X_train, Y_train, batch_size=40, epochs=400, validation_data=(X_val, Y_val)) #train for 400 epochs

mean = 0.0
for _ in range(100): #test the network 100 times and take mean accuracy
    preds = [] #array for predicted outputs
    counter = 0 #counter for correct answers
    predictions = model.predict(X_test) #test
    Y_test = list(Y_test) #correct values 
    for pred in predictions:
        if float(pred) >= 0.50: #binary values 0 or 1 only
            preds.append(int(1))
        elif float(pred) < 0.50:
            preds.append(int(0))


    for i in range(len(preds)):
        if preds[i] == Y_test[i]: #count how many same results, order is same
            counter = counter + 1
    

    mean = mean + counter/len(preds) 



print("Average accuracy " + str((mean/100)*100) + "%")


