from time_lstm import T1TimeLSTM
from dataset_delay_predictions import Dataset_Delay_Prediction
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, Layer, LSTM
import tensorflow as tf

import os
import gzip
import io
import json
import pprint
import datetime
import math
import random
import pickle

experiment = 'VanillaLSTM'
if experiment == 'toy_example':
#Toy example:
    number_of_examples = 12
    example_input_1 = [[[1.0, 0.0, x*1.0] for x in range(10)] for _ in range(number_of_examples)]
    example_input_2 = [[[0.0, 1.0, 2*x*1.0] for x in range(9,-1,-1)] for _ in range(number_of_examples)]
    example_input = example_input_1 + example_input_2

    example_output = [[1.0,0.0] for i in range(number_of_examples)] + [[0.0, 1.0] for i in range(number_of_examples)]

    X_train = np.array(example_input)
    y_train = np.array(example_output)

    print(X_train.shape)
    print(y_train.shape)

    lstm_units = 40
    number_of_epochs = 50

    regressor = Sequential()
    regressor.add(T1TimeLSTM(units=lstm_units))
    regressor.add(Dense(units=y_train.shape[1], activation='softmax'))

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=1, epochs=number_of_epochs)


number_of_event = 60
batch_size=64
data_path = '/home/charles/pconv/data/'
dataset_name = '2019/11/01/hellobank/only-converted-14-days'

pickle_filename = dataset_name.replace('/','-')
if os.path.exists(pickle_filename):
    print('Reading pickle file...')
    dataset = pickle.load(open(pickle_filename, 'rb'))
else:
    dataset = Dataset_Delay_Prediction(dataset_name,
                                        data_path,
                                        number_of_event,
                                        batch_size)
    pickle.dump(dataset, open(pickle_filename, 'wb'))

X_train = dataset.full_features
seqlen = dataset.full_seqlen
y_train = np.array(dataset.full_values)
print(y_train.shape)


lstm_units = 100
number_of_epochs = 10


if experiment=='T1TimeLSTM':
    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print(padded_inputs.shape)
    regressor = Sequential()
    regressor.add(Masking(mask_value=padding_value))
    regressor.add(T1TimeLSTM(units=lstm_units))
    regressor.add(Dense(units=y_train.shape[1], activation='sigmoid'))

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    history = regressor.fit(padded_inputs, y_train, batch_size=1, epochs=number_of_epochs, verbose=2)
    X_test = np.array([padded_inputs[0,:,:]])
    print(X_test.shape)
    a = regressor.predict(X_test)
    print(a)
    print(history.history['loss'])

if experiment == 'VanillaLSTM':
    only_time = False
    if only_time:
        X_train_bis = []
        number_of_event = len(X_train[0][0])
        for idx in range(len(X_train)):
            seq = X_train[idx]
            ts_list = [[a[-1]] for a in seq]
            X_train_bis.append(ts_list)

        padding_value = 0.123456789
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train_bis,
                                                                      padding='post',
                                                                      value=padding_value,
                                                                      dtype='float32')
    else:
        padding_value = 0.123456789
        padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                      padding='post',
                                                                      value=padding_value,
                                                                      dtype='float32')
    print(padded_inputs.shape)
    regressor = Sequential()
    regressor.add(Masking(mask_value=padding_value))
    regressor.add(LSTM(units=lstm_units))
    regressor.add(Dense(units=y_train.shape[1], activation='sigmoid'))

    regressor.compile(optimizer='adam', loss='mean_squared_error')
    history = regressor.fit(padded_inputs, y_train, batch_size=50, epochs=number_of_epochs, verbose=2)
    X_test = np.array([padded_inputs[0, :, :]])
    print(X_test.shape)
    a = regressor.predict(X_test)
    print(a)
    print(history.history['loss'])
