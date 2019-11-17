'''
This is the sampling and sketching task of the assignment. It uses scenario 2
!! To make this code work you have to run this in python2 
'''
import hashlib
import array
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import collections

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(threshold=sys.maxsize)

# Preprocessing data
def preprocess_data():
    data = pd.read_csv('capture20110811.pcap.netflow.labeled', header=0, skiprows=1, delimiter='\s+', names=['date', 'time', 'duration', 'protocol', 'src', 'arrow', 'dst', 'flags', 'tos', 'packets', 'bytes', 'flows', 'label'])
    data = data.iloc[1:]
    data = data.loc[data['label'] != 'Background']
    print(len(data.loc[data['label'] == 'LEGITIMATE']))
    print(len(data.loc[data['label'] == 'Botnet']))
    data['date'] = pd.to_datetime(data['date'])
    data = data.drop('arrow', 1)
    data = data.drop('flows', 1)
    data.protocol, _ = pd.Series(data.protocol).factorize()
    data.flags, _ = pd.Series(data.flags).factorize()
    data['src_ip'], data['src_port'] = data['src'].str.split(':', 1).str
    data['dst_ip'], data['dst_port'] = data['dst'].str.split(':', 1).str
    data = data.drop('dst', 1)
    data = data.drop('src', 1)
    return data

# Code used to perform reservoir sampling.
'''
PARAMETERS
new_data: the data used to perform the sampling on
sample_size: The size of the sample
print_intermedia: Whether or not to print intermediate values while sampling
'''
def sampling(new_data, sample_size, print_intermediate=False):

    # Initialize the reservoir used for sampling
    reservoir = []
    # Start the counter
    N = 0

    # Loop through all data
    for index, row in new_data.iterrows():
        # Print if requested
        if print_intermediate and N % 5000 == 0:
            print(reservoir)

        # Extract source IP
        if row['src_ip'] != '147.32.84.165':
            N += 1
            # Initalize the sample
            if len(reservoir) < sample_size:
                reservoir.append(row['src_ip'])
            else:
                # Given the appropriate chance: m/n, add to sample
                s = int(random.random() * N)
                if s < sample_size:
                    reservoir[s] = row['src_ip']

        # Extract destination IP
        if row['dst_ip'] != '147.32.84.165':
            N += 1
            # Initalize the sample
            if len(reservoir) < sample_size:
                reservoir.append(row['dst_ip'])
            else:
                # Given the appropriate chance: m/n, add to sample
                s = int(random.random() * N)
                if s < sample_size:
                    reservoir[s] = row['dst_ip']


    # Prepare reservoir data to plot
    counter = collections.Counter(reservoir)
    counter = counter.most_common(10)
    keys = [i[0] for i in counter]
    values = [i[1] for i in counter]

    # Plot the data
    plt.barh(range(len(keys)), values)
    plt.yticks(range(len(keys)), keys)
    plt.show()
    return keys

# Code used to create and check the MIN-COUNT sketch.
'''
PARAMETERS
new_data: The data used to perform the sampling on
outcome_amount: The amount of possible values given by the hash functions
hashfunction_amount: The amount of hash functions
checks: The IPs to check.
'''
def mincount(new_data, outcome_amount, hashfunction_amount, checks):
    # Set up the matrix variable
    tables = []
    # Set up the hash functions
    for _ in xrange(hashfunction_amount):
        table = array.array("l", (0 for _ in xrange(outcome_amount)))
        tables.append(table)

    # The function that does the hashing
    def _hash(x):
        md5 = hashlib.md5(str(hash(x)))
        for i in xrange(hashfunction_amount):
            md5.update(str(i))
            # Use modular arithmetic to ensure that outcome_amount is satisfied
            yield int(md5.hexdigest(), 16) % outcome_amount

    # Add an item to the sketch
    def add(x):
        for table, i in zip(tables, _hash(x)):
            table[i] += 1

    # Get the minimum amount of occurrences for a given x
    def query(x):
        return min(table[i] for table, i in zip(tables, _hash(x)))

    # Add the given data to the sketch
    for index, row in new_data.iterrows():
        if row['src_ip'] != '147.32.84.165':
            add(row['src_ip'])
        if row['dst_ip'] != '147.32.84.165':
            add(row['dst_ip'])

    # Check the IPs in the sketch
    for i in checks:
        print(str(i) + "\t" + str(query(i)))

# START OF THE MAIN CODE
print('Preprocessing the data .....')
new_data = preprocess_data()
new_data['label'] = new_data['label'].replace(['Botnet'], 1)
new_data['label'] = new_data['label'].replace(['LEGITIMATE'], 0)

'''TASK 1: SAMPLING'''
# sampling(new_data, 100)
# sampling(new_data, 500)
# sampling(new_data, 1000)
# sampling(new_data, 5000)
# sampling(new_data, 10000)
checks = sampling(new_data, 1000000)

'''TASK 2: SKETCHING'''
mincount(new_data, 20, 20, checks)
mincount(new_data, 200, 200, checks)

