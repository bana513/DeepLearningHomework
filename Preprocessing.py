import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("poloniex_usdt_btc_20170101_DOHLCV_300.csv", sep=';')
print(df.head())
#print(df['close'])

numeric_features = ['date',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume'
                   ]

for feat in numeric_features:
    print("Az outlierek száma a %s tulajdonságban: %d" % (feat, len(df[ abs(df[feat] - np.mean(df[feat])) >3*np.std(df[feat])])))
    #df[feat] = df[feat].apply(lambda x: np.mean(df[feat])+3*np.std(df[feat]) if x > np.mean(df[feat])+3*np.std(df[feat]) else x, 1)
    #df[feat] = df[feat].apply(lambda x: np.mean(df[feat])-3*np.std(df[feat]) if x < np.mean(df[feat])-3*np.std(df[feat]) else x, 1)

plt.plot(range(df.values.shape[0]), df.values[:, 1])
plt.show()

df_values = df.values
timestamps = df_values[:, 0]
dataset = df_values[:, 1:].astype("float")

samples_num = dataset.shape[0]

valid_split = 0.2
test_split = 0.1

dataset_train = dataset[0:int(samples_num * (1 - valid_split - test_split))]
dataset_valid = dataset[int(samples_num * (1 - valid_split - test_split)):int(samples_num * (1 - test_split))]
dataset_test = dataset[int(samples_num * (1 - test_split)):]

print(dataset_train.shape)

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

look_back = 1
X_train, Y_train = create_dataset(dataset_train, look_back)
X_valid, Y_valid = create_dataset(dataset_valid, look_back)
X_test, Y_test = create_dataset(dataset_test, look_back)

X_train = X_train[:, 0, :]
X_valid = X_valid[:, 0, :]
X_test = X_test[:, 0, :]

print(X_train.shape)
print(Y_train.shape)

# Plot 'open' data
plt.plot(dataset_train[:, 0], '-b')
plt.plot([None for i in dataset_train[:, 0]] + [x for x in dataset_valid[:, 0]], '-r')
plt.plot([None for i in dataset_train[:, 0]] + [None for x in dataset_valid[:, 0]] + [x for x in dataset_test[:, 0]], '-k')
plt.show()

# Plot 'high' data
plt.plot(dataset_train[:, 1], '-b')
plt.plot([None for i in dataset_train[:, 1]] + [x for x in dataset_valid[:, 1]], '-r')
plt.plot([None for i in dataset_train[:, 1]] + [None for x in dataset_valid[:, 1]] + [x for x in dataset_test[:, 1]], '-k')
plt.show()

# Plot 'low' data
plt.plot(dataset_train[:, 2], '-b')
plt.plot([None for i in dataset_train[:, 2]] + [x for x in dataset_valid[:, 2]], '-r')
plt.plot([None for i in dataset_train[:, 2]] + [None for x in dataset_valid[:, 2]] + [x for x in dataset_test[:, 2]], '-k')
plt.show()

# Plot 'close' data
plt.plot(dataset_train[:, 3], '-b')
plt.plot([None for i in dataset_train[:, 3]] + [x for x in dataset_valid[:, 3]], '-r')
plt.plot([None for i in dataset_train[:, 3]] + [None for x in dataset_valid[:, 3]] + [x for x in dataset_test[:, 3]], '-k')
plt.show()

# Plot 'volume' data
plt.plot(dataset_train[:, 4], '-b')
plt.plot([None for i in dataset_train[:, 4]] + [x for x in dataset_valid[:, 4]], '-r')
plt.plot([None for i in dataset_train[:, 4]] + [None for x in dataset_valid[:, 4]] + [x for x in dataset_test[:, 4]], '-k')
plt.show()

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
