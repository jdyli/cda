'''
This file is the familiarization task of the data
'''
import pandas as pd 
import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.ar_model import AR
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)

'''LOAD DATASET AND PRE-PROCESS'''
training_set_1 = pd.read_csv('./data/BATADAL_dataset03.csv')
training_set_2 = pd.read_csv('./data/BATADAL_dataset04.csv', header=0)
training_set_2.columns = training_set_2.columns.str.lstrip()
test_set = pd.read_csv('./data/BATADAL_test_dataset.csv', header=0)
test_set.columns = test_set.columns.str.lstrip()

training_set_1['DATETIME'] = training_set_1.DATETIME + ':59:59'
training_set_1['date'] = pd.to_datetime(training_set_1['DATETIME'], format='%d/%m/%y %H:%M:%S')
training_set_2['DATETIME'] = training_set_2.DATETIME + ':59:59'
training_set_2['date'] = pd.to_datetime(training_set_2['DATETIME'], format='%d/%m/%y %H:%M:%S')
training_set_2['ATT_FLAG'] = training_set_2['ATT_FLAG'].replace(to_replace=-999, value=0)
test_set['DATETIME'] = test_set.DATETIME + ':59:59'
test_set['date'] = pd.to_datetime(test_set['DATETIME'], format='%d/%m/%y %H:%M:%S')

# Task 1: Correlation matrix
correlation_matrix = training_set_1.drop('DATETIME', axis=1)
correlation_matrix = correlation_matrix.drop('ATT_FLAG', axis=1)
correlation_matrix = correlation_matrix.drop('date', axis=1)
corr = correlation_matrix.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

# Task 2: Analyzing Cyclical behaviour; use built-in zoom function in the UI of Matplotlib
cyclical_data = training_set_1[['date', 'F_PU1']]
cyclical_data.plot(x='date')
plt.title('Cyclical behaviour of F_PU1')
plt.show()

# Task 3: First initial prediction with use of an autogressive model; use built-in zoom function in the UI of Matplotlib 
cyclical_data.set_index('date', inplace=True)
train_size = int(len(cyclical_data) * 0.7)
train, test = cyclical_data[0:train_size], cyclical_data[train_size:len(cyclical_data)]
model = AR(train, freq='H')
model_fitted = model.fit()
predictions = model_fitted.predict(
    start=len(train),
    end=len(train) + len(test) - 1,
    dynamic=False
)
comparison = pd.concat([cyclical_data['F_PU1'], predictions], axis=1).rename(columns={'pump': 'actual', 0: 'predicted'})
comparison.plot()
plt.title('Forecast of F_PU1 with AR model')
plt.show()