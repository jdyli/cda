'''
This is the profiling task. It uses scenario 10 as dataset

You can use the following commands if the libraries are not downloaded yet
pip install hmmlearn
pip install tqdm
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from hmmlearn.hmm import GaussianHMM
import sys
import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import warnings
from discretization import find_optimal_clusters, cluster_values, netflow_encoding

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

# Perform the encoding of the data
'''
PARAMETERS
column_names: the column names to consider for the encoding
host: the netflow data
dataset: name of the dataset
'''
def perform_encoding(column_names, host, dataset):
    column_list = [value for key,value in column_names.items()]
    spacesize = column_list[0]
    for i in column_list[1:len(column_list)]:
        spacesize = spacesize * i
    print('Discretizing the ', dataset, '.....')
    for i in tqdm.tqdm(range(0, len(host))):
        host.at[i, 'code'] = netflow_encoding(host.iloc[i], column_names, spacesize, column_list)
    return host

# Partitions the data into windows
'''
PARAMETERS
data: the dataset to perform the sliding
n: window size
'''
def sliding_window(data, n=20):
    new_features = []
    for i in range(len(data)-n):
        new_features.append(data[i:i+n,])
    new_features = np.array(new_features)
    return(new_features)

# Builds/ trains the Gaussian Hidden Markov Model
'''
PARAMETERS
data: the infected host that is used to build the model
'''
def build_model(data, columns):
    features_train = np.float32(data[columns].as_matrix())
    discrete_features_train = sliding_window(features_train)
    model = GaussianHMM(n_components=3)
    model.fit(discrete_features_train)
    return (model, model.decode(discrete_features_train)[1])

# Calculates the error threshold to determine whether a host is infectious or benign
'''
PARAMETERS
unique_hosts: set of unique that will be evaluated
model: the trained Gaussian HMM model
data: the total dataset
benign_state: the calculated sequence of the states based on the Gaussian HMM model
columns: the columns to consider for the model
'''
def determine_threshold(unique_hosts, model, data, benign_state, columns):
    average = []
    average_error = []
    for host in unique_hosts:
        host_data = data.loc[data['src_ip'] == host]
        host_features = np.float32(host_data[columns].as_matrix())
        host_discrete = sliding_window(host_features)
        if (len(host_discrete) == 0):
            average.append(0)
            average_error.append(0)
            continue
        prediction = np.mean(model.predict(host_discrete))
        error = abs((np.mean(benign_state) - np.mean(prediction)))
        average_error.append(error)
        average.append(np.mean(prediction))
    threshold = np.mean(average_error) + np.std(average_error)
    return threshold

# Predicts for unseen data whether a host is infectious or benign
'''
PARAMETERS
unique_hosts: set of unique hosts that will be evaluated
model: the Gaussian HMM model that has been build 
data: the total dataset
threshold: the determined error threshold to evaluate if a host is infectious 
state: the sequence state of the built Gaussian HMM
columns: the columns to consider for the model
'''
def predict(unique_hosts, model, data, threshold, state, columns):
    predict = []
    label = []
    for host in tqdm.tqdm(range(0, len(unique_hosts))):
        host_data = data.loc[data['src_ip'] == unique_hosts[host]]
        host_features = np.float32(host_data[columns].as_matrix())
        host_discrete = sliding_window(host_features)
        if (len(host_discrete) == 0):
            predict.append(0)
            label.append(0)
            continue
        h_predict = 0 if (np.mean(state) - np.mean(model.predict(host_discrete))) > threshold else 1
        h_label = 0 if host_data['label'].unique()[0] == 0 else 1
        predict.append(h_predict)
        label.append(h_label)
    print('Confusion matrix: ')
    print(confusion_matrix(label, predict))


print('Preprocessing the data .....')
new_data = preprocess_data()

# Creating the three different datasets
benign_set = new_data.loc[new_data['label'] == 0]
configuration_set = benign_set[(benign_set.index<=np.percentile(benign_set.index, 30))]
training_host = new_data.loc[new_data['src_ip'] == '147.32.84.209']
validation_set = new_data.drop(configuration_set.index)
validation_set = validation_set.drop(training_host.index)
configuration_set = configuration_set.reset_index(drop=True)
training_host = training_host.reset_index(drop=True)
validation_set = validation_set.reset_index(drop=True)
print('Size of configuration set: ', len(configuration_set))
print('Size of training set: ', len(training_host))
print('Size of validation set: ', len(validation_set))

# Determine number of clusters and error threshold based on configuration set
'''print('Calculating the clusters and plotting the ELBOW for the features .....')
find_optimal_clusters(configuration_set['duration'], 'Duration') # 5 clusters
find_optimal_clusters(configuration_set['bytes'], 'Bytes') # 12 clusters
find_optimal_clusters(configuration_set['packets'], 'Packets') # 7 clusters
'''
configuration_set = cluster_values(configuration_set, 5, 'duration') # The clusters were already determined, so running the above commented code is optional
configuration_set = cluster_values(configuration_set, 12, 'bytes')
configuration_set = cluster_values(configuration_set, 7, 'packets')
column_names = {'protocol': len(configuration_set['protocol'].unique()), 'flags': len(configuration_set['flags'].unique()), 
                    'duration_cluster': len(configuration_set['duration_cluster'].unique()), 'bytes_cluster': len(configuration_set['bytes_cluster'].unique()),
                    'packets_cluster': len(configuration_set['packets_cluster'].unique())}
configuration_set = perform_encoding(column_names, configuration_set, 'Configuration set')

print('Determining the threshold based on the configuration set .....')
host_frequency_list = configuration_set['src_ip'].value_counts()[:25]
train_configuration_host = configuration_set.loc[configuration_set['src_ip'] == host_frequency_list.index[0]]
(model, benign_state) = build_model(train_configuration_host, 'code')
threshold = determine_threshold(host_frequency_list.index[1:], model, configuration_set, benign_state, 'code')

# Train set and validation set. Prepare the train- and validation set
training_host = cluster_values(training_host, 5, 'duration')
training_host = cluster_values(training_host, 12, 'bytes')
training_host = cluster_values(training_host, 7, 'packets')
column_names_train = {'protocol': len(training_host['protocol'].unique()), 'flags': len(training_host['flags'].unique()), 
                    'duration_cluster': len(training_host['duration_cluster'].unique()), 'bytes_cluster': len(training_host['bytes_cluster'].unique()),
                    'packets_cluster': len(training_host['packets_cluster'].unique())}
training_host = perform_encoding(column_names, training_host, 'training set')

# Discretization on validation set
validation_set = cluster_values(validation_set, 5, 'duration')
validation_set = cluster_values(validation_set, 12, 'bytes')
validation_set = cluster_values(validation_set, 7, 'packets')
column_names_validation = {'protocol': len(validation_set['protocol'].unique()), 'flags': len(validation_set['flags'].unique()), 
                    'duration_cluster': len(validation_set['duration_cluster'].unique()), 'bytes_cluster': len(validation_set['bytes_cluster'].unique()),
                    'packets_cluster': len(validation_set['packets_cluster'].unique())}
validation_set = perform_encoding(column_names_validation, validation_set, 'validation set')

# Compute the frequency of netflows for each host in validation set
host_frequency = validation_set['src_ip'].value_counts()
host_frequency_subset = host_frequency[:380]
host_frequency_subset.plot()
plt.xlabel("Host")
plt.ylabel("Frequencies")
plt.show()
print('Calculating the number of infected hosts on the validation set .....')
validation_unique_hosts = host_frequency_subset.index
train_model, infected_host = build_model(training_host, 'code')
predict(validation_unique_hosts, train_model, validation_set, threshold, infected_host, 'code')
