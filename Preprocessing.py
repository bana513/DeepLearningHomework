import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Import data from csv file into a pandas dataframe and check that if it is correct
df = pd.read_csv("poloniex_usdt_btc_20170101_DOHLCV_300.csv", sep=';')
print(df.head())

# Plot 'open' values
plt.plot(range(df.values.shape[0]), df.values[:, 1])
plt.show()

# Turn dataframe into a numpy floating point array
df_values = df.values
timestamps = df_values[:, 0]
dataset = df_values[:, 1:].astype("float")


# Split the dataset into training (70%), cross validation (20%) and test (10%) set
samples_num = dataset.shape[0]

valid_split = 0.2
test_split = 0.1

dataset_train = dataset[0:int(samples_num * (1 - valid_split - test_split))]
dataset_valid = dataset[int(samples_num * (1 - valid_split - test_split)):int(samples_num * (1 - test_split))]
dataset_test = dataset[int(samples_num * (1 - test_split)):]


# Helper function for creating dataset for LSTM components
# [Description:   We want to predict the next value in time, so we have to
#                 shift the values with the 'look_back' variable]
def create_dataset_for_lstm(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# Call the helper function for the training, cross validation and test sets,
# then format them consistently
look_back = 1
X_train, Y_train = create_dataset_for_lstm(dataset_train, look_back)
X_valid, Y_valid = create_dataset_for_lstm(dataset_valid, look_back)
X_test, Y_test = create_dataset_for_lstm(dataset_test, look_back)

X_train = X_train[:, 0, :]
X_valid = X_valid[:, 0, :]
X_test = X_test[:, 0, :]

print(X_train.shape)
print(Y_train.shape)

# Plot 'open' dataset
plt.plot(dataset_train[:, 0], '-b')
plt.plot([None for i in dataset_train[:, 0]] + [x for x in dataset_valid[:, 0]], '-r')
plt.plot([None for i in dataset_train[:, 0]] + [None for x in dataset_valid[:, 0]] + [x for x in dataset_test[:, 0]], '-k')
plt.show()

# Plot 'high' dataset
plt.plot(dataset_train[:, 1], '-b')
plt.plot([None for i in dataset_train[:, 1]] + [x for x in dataset_valid[:, 1]], '-r')
plt.plot([None for i in dataset_train[:, 1]] + [None for x in dataset_valid[:, 1]] + [x for x in dataset_test[:, 1]], '-k')
plt.show()

# Plot 'low' dataset
plt.plot(dataset_train[:, 2], '-b')
plt.plot([None for i in dataset_train[:, 2]] + [x for x in dataset_valid[:, 2]], '-r')
plt.plot([None for i in dataset_train[:, 2]] + [None for x in dataset_valid[:, 2]] + [x for x in dataset_test[:, 2]], '-k')
plt.show()

# Plot 'close' dataset
plt.plot(dataset_train[:, 3], '-b')
plt.plot([None for i in dataset_train[:, 3]] + [x for x in dataset_valid[:, 3]], '-r')
plt.plot([None for i in dataset_train[:, 3]] + [None for x in dataset_valid[:, 3]] + [x for x in dataset_test[:, 3]], '-k')
plt.show()

# Plot 'volume' dataset
plt.plot(dataset_train[:, 4], '-b')
plt.plot([None for i in dataset_train[:, 4]] + [x for x in dataset_valid[:, 4]], '-r')
plt.plot([None for i in dataset_train[:, 4]] + [None for x in dataset_valid[:, 4]] + [x for x in dataset_test[:, 4]], '-k')
plt.show()

# Normalize data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print (X_train)