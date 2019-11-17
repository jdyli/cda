'''
This file is the classification task of the netflows. It uses scenario 10 as dataset
'''

import pandas as pd
from sklearn import metrics
import numpy as np
import sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import svm
from imblearn.over_sampling import SMOTE
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=sys.maxsize)

# Preprocessing data
def preprocess_data():
    data = pd.read_csv('capture20110818.pcap.netflow.labeled', header=0, skiprows=1, delimiter='\s+', names=['date', 'time', 'duration', 'protocol', 'src', 'arrow', 'dst', 'flags', 'tos', 'packets', 'bytes', 'flows', 'label'])
    data = data.iloc[1:]
    data = data.loc[data['label'] != 'Background']
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop('arrow', 1)
    data = data.drop('flows', 1)
    data.protocol, _ = pd.Series(data.protocol).factorize()
    data.flags, _ = pd.Series(data.flags).factorize()
    data['src_ip'], data['src_port'] = data['src'].str.split(':', 1).str
    data['dst_ip'], data['dst_port'] = data['dst'].str.split(':', 1).str
    data = data.drop('dst', 1)
    data = data.drop('src', 1)
    data['label'] = data['label'].replace(['Botnet'], 1)
    data['label'] = data['label'].replace(['LEGITIMATE'], 0)
    data['code'] = 0
    return data

print('Preprocessing the data .....')
drop_tables = ['label', 'date', 'src_ip', 'dst_ip', 'time', 'src_port', 'dst_port', 'code']
new_data = preprocess_data()
unique_hosts = new_data['src_ip'].unique()
new_data['label'] = new_data['label'].replace(['Botnet'], 1)
new_data['label'] = new_data['label'].replace(['LEGITIMATE'], 0)

# Creating the three different datasets
benign_set = new_data.loc[new_data['label'] == 0]
configuration_set = benign_set[(benign_set.index<=np.percentile(benign_set.index, 30))]
infected_host = new_data.loc[(new_data['src_ip'] == '147.32.84.209')]
big_test_set = new_data.drop(configuration_set.index)
big_test_set = big_test_set.drop(infected_host.index)

new_data = infected_host.append(benign_set, ignore_index=True) # Combine benign hosts and infected host for training set
new_data = shuffle(new_data) #Shuffle the data to make the data not time-dependent
y = new_data['label']
X = new_data.drop(drop_tables, 1)
host_frequency = big_test_set['src_ip'].value_counts() # Take top 380 hosts as test set
host_frequency_subset = host_frequency[:380]
criteria = big_test_set['src_ip'].isin(host_frequency_subset.index)
big_test_set = big_test_set[criteria]
y_big_test_set = big_test_set['label']
src_big_test_set = big_test_set['src_ip']
x_big_test_set = big_test_set.drop(drop_tables, 1)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35) #train/ validation
sm = SMOTE(ratio = 1)
x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train) # SMOTE-ed training data

print("\nBefore SMOTE, counts of label 'Botnet': {}".format(sum(y_train==1)))
print("Before SMOTE, counts of label 'LEGITIMATE': {} \n".format(sum(y_train==0)))
print("\nAfter SMOTE, counts of label 'Botnet': {}".format(sum(y_train_sm==1)))
print("After SMOTE, counts of label 'LEGITIMATE': {} \n".format(sum(y_train_sm==0)))
print("\nValidation data, counts of label 'Botnet': {}".format(sum(y_test==1)))
print("Validation data, counts of label 'LEGITIMATE': {} \n".format(sum(y_test==0)))
print("\nTest data, counts of label 'Botnet': {}".format(sum(y_big_test_set==1)))
print("Test data, counts of label 'LEGITIMATE': {} \n".format(sum(y_big_test_set==0)))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_sm, y_train_sm)

print('Predicting on the validation set. This can take a while .....')
predict = neigh.predict(x_test)
print('Confusion matrix on validation set: ')
print(confusion_matrix(y_test, predict))

# Evaluate on packet-level
print('Predicting on the test set. This can take a while .....')
predict_big = neigh.predict(x_big_test_set)
print('Confusion matrix on test set: ')
print(confusion_matrix(y_big_test_set, predict_big))
x_big_test_set['predicted_label'] = predict_big
x_big_test_set['actual_label'] = y_big_test_set

# Evaluate on host-level
x_big_test_set['src_ip'] = src_big_test_set
mean_col = x_big_test_set.groupby(['src_ip'])['predicted_label'].mean()
mean_col2 = x_big_test_set.groupby(['src_ip'])['actual_label'].mean()
mean_col = [1 if x > 0 else 0 for x in mean_col]
mean_col2 = [1 if x > 0 else 0 for x in mean_col2]
print('Confusion matrix on host-level')
print(confusion_matrix(mean_col2, mean_col))
