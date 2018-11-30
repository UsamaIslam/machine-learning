import Assignment_1_KNN.KNN_q as kn
import pandas as pd
import numpy as np


def root_mean_square_error(actual, predictions):
    return (((predictions - actual) ** 2).sum()) / len(predictions)


test_df = pd.read_csv('data/kc_house_test_data.csv', usecols=[i for i in range(3, 21)], header=None, index_col=False)
# print(test_df.values)
test_df = list(test_df.values)
# print(test_df[1::])

test_df = test_df[1:1000:]
lable_test = pd.read_csv('data/kc_house_test_data.csv', usecols=[2, ], header=None, index_col=False)
lable_test = list(lable_test.values)
lable_test = lable_test[1:1000:]
train_df = pd.read_csv('data/kc_house_train_data.csv', usecols=[i for i in range(3, 21)], header=None, index_col=False)
train_df = list(train_df.values)
train_df = train_df[1:2000:]
lable_train = pd.read_csv('data/kc_house_train_data.csv', usecols=[2, ], header=None, index_col=False)
lable_train = list(lable_train.values)
lable_train = lable_train[1:2000:]

predictions = []
for testInstance in test_df:
    predictions.append(kn.getResponse(kn.getNeighbors(train_df, testInstance, lable_train, k=5)))
rmse = root_mean_square_error(np.asarray(lable_test, dtype='float64'), np.asarray(predictions, dtype='float64'))
print(rmse)
