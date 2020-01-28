from time_lstm import T1TimeLSTM, T2TimeLSTM
from dataset_delay_predictions import Dataset_Delay_Prediction, Dataset_Delay_Prediction_from_list, DatasetDelayPredictionStackOverflow
from time_dependant_representation import TimeDepMasking, TimeDepJointEmbedding
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Masking, Layer, LSTM
import tensorflow as tf
import os
import pickle
from matplotlib import pyplot as plt


methods_list = ['VanillaLSTMUni',
                'VanillaLSTMMulti',
                'T1TimeLSTM',
                'T2TimeLSTM',
                'TimeDepJointEmbedding',
                #'PhasedLSTM',
                'Naive']


dataset_names = [
    '2019-10_11_12-dataset_delay_prediction',
    'stack-overflow-dataset'
]




dataset_name = dataset_names[1] #experience on Stack Overflow

pickle_filename = dataset_name.replace('/', '-')
if os.path.exists(pickle_filename):
    print('Reading pickle file...')
    dataset = pickle.load(open(pickle_filename, 'rb'))


X_train = dataset.full_features_dt[:dataset.training_set_length]
seqlen = dataset.full_seqlen[:dataset.training_set_length]
y_train = np.array(dataset.full_values[:dataset.training_set_length])
print(y_train.shape)

X_test = dataset.full_features_dt[dataset.training_set_length:]
seqlen_test = dataset.full_seqlen[dataset.training_set_length:]
y_test = np.array(dataset.full_values[dataset.training_set_length:])

test_set_length = len(y_test)

lstm_units = 50
number_of_epochs = 50

history_dict = {}
MSE = {}

number_of_events = dataset.number_of_events
method = methods_list[0]
if method == 'VanillaLSTMUni':
    # We only consider time here, not the value of the event

    X_train_bis = []
    for idx in range(len(X_train)):
        seq = X_train[idx]
        ts_list = [[a[-1]] for a in seq]
        X_train_bis.append(ts_list)

    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train_bis,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print('method: ', method)
    print('input shape :', str(padded_inputs.shape))

    regressor_LSTMUni = Sequential()
    regressor_LSTMUni.add(Masking(mask_value=padding_value))
    regressor_LSTMUni.add(LSTM(units=lstm_units))
    regressor_LSTMUni.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    regressor_LSTMUni.compile(optimizer='adam', loss='mean_squared_error')
    history_dict[method] = regressor_LSTMUni.fit(padded_inputs, y_train, batch_size=50, epochs=number_of_epochs, verbose=2)
    print('Training done for method: ', method)

    X_test_bis = []
    for idx in range(test_set_length):
        seq = X_test[idx]
        ts_list = [[a[-1]] for a in seq]
        X_test_bis.append(ts_list)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_test_bis,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')
    pred = regressor_LSTMUni.predict(padded_inputs)
    mean_sum_of_squares = sum([(y_test[i] - pred[i]) ** 2 for i in range(test_set_length)]) / test_set_length
    MSE[method] = mean_sum_of_squares

if method == 'VanillaLSTMMulti':
    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print('method: ', method)
    print('input shape :', str(padded_inputs.shape))

    regressor_LSTMMulti = Sequential()
    regressor_LSTMMulti.add(Masking(mask_value=padding_value))
    regressor_LSTMMulti.add(LSTM(units=lstm_units))
    regressor_LSTMMulti.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    regressor_LSTMMulti.compile(optimizer='adam', loss='mean_squared_error')
    history_dict[method] = regressor_LSTMMulti.fit(padded_inputs, y_train, batch_size=50, epochs=number_of_epochs,
                                                 verbose=2)
    print('Training done for method: ', method)


    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')
    pred = regressor_LSTMMulti.predict(padded_inputs)
    mean_sum_of_squares = sum([(y_test[i] - pred[i]) ** 2 for i in range(test_set_length)]) / test_set_length
    MSE[method] = mean_sum_of_squares

if method == 'T1TimeLSTM':
    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print('method: ', method)
    print('input shape :', str(padded_inputs.shape))

    regressor_T1TimeLSTM = Sequential()
    regressor_T1TimeLSTM.add(Masking(mask_value=padding_value))
    regressor_T1TimeLSTM.add(LSTM(units=lstm_units))
    regressor_T1TimeLSTM.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    regressor_T1TimeLSTM.compile(optimizer='adam', loss='mean_squared_error')
    history_dict[method] = regressor_T1TimeLSTM.fit(padded_inputs, y_train, batch_size=1, epochs=number_of_epochs,
                                                   verbose=2)
    print('Training done for method: ', method)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')
    pred = regressor_T1TimeLSTM.predict(padded_inputs)
    mean_sum_of_squares = sum([(y_test[i] - pred[i]) ** 2 for i in range(test_set_length)]) / test_set_length
    MSE[method] = mean_sum_of_squares

if method == 'T2TimeLSTM':
    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print('method: ', method)
    print('input shape :', str(padded_inputs.shape))

    regressor_T2TimeLSTM = Sequential()
    regressor_T2TimeLSTM.add(Masking(mask_value=padding_value))
    regressor_T2TimeLSTM.add(LSTM(units=lstm_units))
    regressor_T2TimeLSTM.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    regressor_T2TimeLSTM.compile(optimizer='adam', loss='mean_squared_error')
    history_dict[method] = regressor_T2TimeLSTM.fit(padded_inputs, y_train, batch_size=1, epochs=number_of_epochs,
                                                   verbose=2)
    print('Training done for method: ', method)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')
    pred = regressor_T2TimeLSTM.predict(padded_inputs)
    mean_sum_of_squares = sum([(y_test[i] - pred[i]) ** 2 for i in range(test_set_length)]) / test_set_length
    MSE[method] = mean_sum_of_squares

if method == 'TimeDepJointEmbedding':
    padding_value = 0.123456789
    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_train,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')

    print('method: ', method)
    print('input shape :', str(padded_inputs.shape))

    regressor_TimeDep = Sequential()
    regressor_TimeDep.add(Masking(mask_value=padding_value))
    regressor_TimeDep.add(LSTM(units=number_of_events))
    regressor_TimeDep.add(Dense(units=y_train.shape[1], activation='sigmoid'))
    regressor_TimeDep.compile(optimizer='adam', loss='mean_squared_error')
    history_dict[method] = regressor_TimeDep.fit(padded_inputs, y_train, batch_size=1, epochs=number_of_epochs,
                                                    verbose=2)
    print('Training done for method: ', method)

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(X_test,
                                                                  padding='post',
                                                                  value=padding_value,
                                                                  dtype='float32')
    pred = regressor_TimeDep.predict(padded_inputs)
    mean_sum_of_squares = sum([(y_test[i] - pred[i]) ** 2 for i in range(test_set_length)]) / test_set_length
    MSE[method] = mean_sum_of_squares