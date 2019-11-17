'''
This is the discretization task inspired by the paper "Learning behavioral fingerprints from netflows using timed automata"
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import tqdm 

# Preprocessing data
def preprocess_data():
    data = pd.read_csv('capture20110818.pcap.netflow.labeled', header=0, skiprows=1, delimiter='\s+', names=['date', 'time', 'duration', 'protocol', 'src', 'arrow', 'dst', 'flags', 'tos', 'packets', 'bytes', 'flows', 'label'])
    data = data.iloc[1:]
    data = data.loc[data['label'] != 'Background']
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop('arrow', 1)
    data = data.drop('flows', 1)
    data.protocol, mapping = pd.Series(data.protocol).factorize()
    data.flags, _ = pd.Series(data.flags).factorize()
    data['src_ip'], data['src_port'] = data['src'].str.split(':', 1).str
    data['dst_ip'], data['dst_port'] = data['dst'].str.split(':', 1).str
    data = data.drop('dst', 1)
    data = data.drop('src', 1)
    data['label'] = data['label'].replace(['Botnet'], 1)
    data['label'] = data['label'].replace(['LEGITIMATE'], 0)
    data['code'] = 0
    return data, mapping

# Finds optimal number of clusters for numeric-valued attributes
'''
PARAMETERS
data: the numeric-valued attribute to determine the number of clusters
title: name of the attribute
'''
def find_optimal_clusters(data, title):
    X = data.values.reshape(-1,1)
    distortions = []
    K = range(1,30)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k for attribute ' + title)
    plt.show()

# Based on the found clusters, the numeric values will be converted into numeric categories
'''
PARAMETERS
data: the used attribute to perform conversion from numeric into numeric categories
n_clusters: number of clusters based on find_optimal_cluster()
title: the name of the attribute
'''
def cluster_values(data, n_clusters, title):
    N = len(data)
    p = round(1 / n_clusters * N)
    criteria = [data[title].between((i*p)+1, p*(i+1)) for i in range(0, n_clusters)]
    values = [i for i in range(1, n_clusters+1)]
    data[title+'_cluster'] = np.select(criteria, values, 0)
    return data

# Performs the encoding of the attributes
'''
PARAMETERS
netflow: the netflow to perform the encoding
column_names: the columns to be considered
spacesize: the initial spacesize which is based on the unique values of each attribute
column_list: the value of the columns to consider
'''
def netflow_encoding(netflow, column_names, spacesize, column_list):
    names = list(column_names.keys())
    code = 0
    for i in range(0, len(column_list)):
        code = code + (netflow[names[i]] * spacesize / column_list[i])
        spacesize = spacesize / column_list[i]
    return code

# Performs visualizations of the two chosen features
'''
PARAMETERS
new_data: the data used for the visualizations
'''
def visualization(new_data):
    legitimate = new_data[new_data['label'] == 0]
    botnet = new_data[new_data['label'] == 1]

    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = [4, 8]

    legitimate.boxplot(column=['bytes'])
    plt.title('Benign hosts')
    plt.show()
    botnet.boxplot(column=['bytes'])
    plt.title('Infected hosts')
    plt.show()

    protocol_counts_legitimate = legitimate['protocol'].value_counts()
    plt.bar(range(len(protocol_counts_legitimate)), protocol_counts_legitimate.values, align='center')
    plt.xticks(range(len(protocol_counts_legitimate)), protocol_counts_legitimate.index.values, size='small')
    plt.title('Benign hosts') 
    plt.show()

    protocol_counts_botnet = botnet['protocol'].value_counts() 
    plt.bar(range(len(protocol_counts_botnet)), protocol_counts_botnet.values, align='center') 
    plt.xticks(range(len(protocol_counts_botnet)), protocol_counts_botnet.index.values, size='small') 
    plt.title('Infected hosts')
    plt.show()  
    
def main():
    print('Preprocessing the data .....')
    (new_data, mapping) = preprocess_data()
    visualization(new_data)
    infected_host = new_data.loc[new_data['src_ip'] == '147.32.84.209']
    infected_host = infected_host.reset_index(drop=True)

    # Find optimal number of clusters for attribute Bytes
    find_optimal_clusters(infected_host['bytes'], 'Bytes') # 4 clusters
    infected_host = cluster_values(infected_host, 4, 'bytes')
    column_names = {'protocol': len(infected_host['protocol'].unique()), 'bytes_cluster': len(infected_host['bytes_cluster'].unique())}

    # Perform the encoding on infected host
    column_list = [value for key,value in column_names.items()]
    spacesize = column_list[0]
    for i in column_list[1:len(column_list)]:
        spacesize = spacesize * i
    print('Discretizing the infected host .....')
    for i in tqdm.tqdm(range(0, len(infected_host))):
        infected_host.at[i, 'code'] = netflow_encoding(infected_host.iloc[i], column_names, spacesize, column_list)

    '''APPLY ON ALL OTHER HOSTS'''
    other_hosts = new_data.loc[new_data['src_ip'] != '147.32.84.209']
    other_hosts = other_hosts.reset_index(drop=True)
    other_hosts = cluster_values(other_hosts, 4, 'bytes')
    column_names_other = {'protocol': len(other_hosts['protocol'].unique()), 'bytes_cluster': len(other_hosts['bytes_cluster'].unique())}

    # Perform the encoding on other hosts
    column_list = [value for key,value in column_names_other.items()]
    spacesize = column_list[0]
    for i in column_list[1:len(column_list)]:
        spacesize = spacesize * i
    print('Discretizing the other hosts .....')
    for i in tqdm.tqdm(range(0, len(other_hosts))):
        other_hosts.at[i, 'code'] = netflow_encoding(other_hosts.iloc[i], column_names, spacesize, column_list)

    # Analyze the differences between benign and infected hosts
    other_infected_hosts = other_hosts.loc[other_hosts['label'] == 1]
    benign_hosts = other_hosts.loc[other_hosts['label'] == 0]

    print('Distribution of the code feature ....')
    print(other_infected_hosts['code'].value_counts())
    print(benign_hosts['code'].value_counts())
    print(infected_host['code'].value_counts())

if __name__ == "__main__":
    main()
