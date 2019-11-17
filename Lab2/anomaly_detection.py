'''
Note for the ARMA task: to find the optimal p and q parameter values we used the library pmdarima. 
Since this might take a while to complete we have saved the optimal values of the parameters
for you in a dictionary (line 145). This way you do not have to run the complete search. However,
if you would like to run the complete search then that is also possible. Upon running the code you will
be asked which one you like to execute.

If you do not have the correct libraries yet, you can use the commands below
to install the correct packages via pip.

pip install pandas
pip install numpy
pip install seaborn
pip install matplotlib
pip install statsmodels
pip install pmdarima
pip install saxpy

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pmdarima as pm
from saxpy.sax import sax_via_window
from saxpy.hotsax import find_discords_hotsax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import warnings

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)

# Plot the (partial) autocorrelation of the data
'''
PARAMETERS
data: the dataset that contains the features are used for the (partial) correlation plots
lags=24: 24 is chosen because it showed the plots visually understandably and the data has hourly datetime
'''
def plot_autocorrelation(data):
    for i in data:
        print('Feature: ', i)
        plot_acf(train_data[i], lags=24)
        plt.show()
        plot_pacf(train_data[i], lags=24)
        plt.show()

# Use the computed p, q for the grid search
'''
PARAMETERS
data: the data where the ARMA model is fit on
p_value: the maximum value of p determined by the a
q_value: the maximum value of q
'''
def find_optimal_parameters(data, p_value, q_value):
    model = pm.auto_arima(data, start_p=1, start_q=1,
                        max_p=p_value, max_q=q_value,   # maximum p and q
                        test='adf',                     # use ADF test that handles time series well
                        m=1,                            # frequency of series
                        trace=True,                     # print the status of the fits
                        error_action='ignore',          # when no AIC is found can be ignored, go to the next
                        suppress_warnings=True)         # internal warnings of statsmodel can be ignored
    print(model.summary())
    return model

# Create ARMA model with optimal p,q and fit + predict on the data
'''
PARAMETERS
data: the data where the ARMA model is fit on
order_value: (p, d, q) the p and q value for the model. This is optimized with the use of find_optimal_parameters. d is set 
            to 0 because for this assignment we can create an ARMA model and not an ARIMA model
threshold: the threshold for the anomaly
'''
def build_and_predict_model(data, order_value, threshold):
    print('Using the parameter order: ', order_value)
    model = ARIMA(data['actual'], order=order_value, freq='H')
    model_fit = model.fit(disp=0)
    prediction = model_fit.predict(dynamic=False)
    data['predictions'] = pd.DataFrame(data=prediction).values
    data['residual'] = abs(data['predictions'] - data['actual'])
    data['mean'] = data['residual'].mean()
    data['threshold'] = threshold
    data['anomalies'] = 0
    data.loc[data['residual'] > data['threshold'], 'anomalies'] = 1
    return data

# Visualize the ARMA model
'''
PARAMETERS
data: the data where the ARMA model is fit on
is_attackset: boolean variable whether the data contains labels with attacks or not
feature_name: the feature name of the designated model
'''
def visualize_model(data, is_attackset, feature_name):
    plt.figure(figsize=(12,5), dpi=100)
    if (is_attackset):
        plt.plot(data['residual'], label='Residual error')
        if ('label' in data):
            plt.plot(data['label'], label='Attack')
            plt.title('Residual error of ' + feature_name + ' with attacks')
        plt.plot(data['mean'], label='Mean')
        plt.plot(data['threshold'], '--', label='Threshold')
        plt.plot(data['residual'].loc[data['anomalies'] == 1], 'ro', label='Anomalies')
        
    else:
        plt.plot(data['actual'], label='Actual')
        plt.plot(data['predictions'], label='Prediction')
        plt.title('Predictions vs Actuals of ' + feature_name + ' on normal training set')
    plt.legend(loc='upper right', fontsize=8)
    plt.show()

# Discrete model: finds anomalies using a discrete model analysis
'''
PARAMETERS
ano_set: The dataset on which anomalies will be detected
train_set: The set used to train the anomaly model.
anomaly_set: the set used to label the dates that are found to be anomalous
win_size: The window size of the sliding windows
paa_size: The length of the sequences that will be found
alphabet_size: The amount of unique letters used in the sequence
signal: The signal on which anomalies will be detected
'''
def discretization(ano_set, train_set, anomaly_set, win_size, paa_size, alphabet_size, signal):
    print('Computing discrete model for ', signal, ' .......')

    # Find discrete sequences in the training data
    sax1 = sax_via_window(series=np.array(train_set[signal]), win_size=win_size, paa_size=paa_size,
                          alphabet_size=alphabet_size,
                          nr_strategy='exact', z_threshold=0.01)

    # Find discrete sequences in the anomalous data
    sax2 = sax_via_window(series=np.array(ano_set[signal]), win_size=win_size, paa_size=paa_size,
                          alphabet_size=alphabet_size,
                          nr_strategy='exact', z_threshold=0.01)

    # List discrete sequences that are in the anomaly set, but not in the training set
    attacklist = []
    for k, v in sax2.items():
        if k not in sax1:
            for e in v:
                attacklist.append(ano_set.index[e])

    # Sort the dates that are found and remove None values
    attacklist = [i for i in attacklist if i is not None]
    attacklist.sort()

    # Add the found anomalies to the output and return
    for i in attacklist:
        anomaly_set.at[i, 'anomaly'] = 1
    return anomaly_set

# PCA model
'''
PARAMETERS
test_set: The dataset on which anomalies will be detected
train_set: The set used to train the anomaly model.
anomaly_set: the set used to label the dates that are found to be anomalous
'''
def pca(train_set, test_set, anomaly_set):
    print('Computing PCA ....')
    # Preprocess the data; remove notions of time for PCA and normalize the data
    no_dates_test_set = test_set.drop(columns=['DATETIME', 'ATT_FLAG'])
    normalized_df = (train_set.drop(columns=['DATETIME', 'date', 'ATT_FLAG']) - train_set.drop(columns=['DATETIME', 'date', 'ATT_FLAG']).mean())
    pca_model = PCA().fit(normalized_df)

    # Plot the variances
    plt.plot(pca_model.explained_variance_ratio_) # Explained percentage of variance explained by each selected components
    plt.xlabel('Principal components')
    plt.ylabel('Explained variance')
    plt.show()

    pca_model = PCA(n_components=10, svd_solver='full')
    pca_model.fit(normalized_df)
    training_pca = pca_model.transform(normalized_df)

    # calculate the residuals
    inverse_training_pca = pca_model.inverse_transform(training_pca)
    residuals = normalized_df - inverse_training_pca
    residuals = np.abs(residuals.sum(axis = 1, skipna = True))

    # plot the residuals
    plt.plot(residuals)
    plt.title('Residuals on unfiltered training data')
    plt.show()

    # filter the residual peaks
    filtered_set = []
    for i in range(0, len(residuals)):
        if residuals[i] < 10:
            filtered_set.append(normalized_df.loc[i])

    # fit with filtered data
    pca_model.fit(filtered_set)

    # create residual graph with new filtered training data
    training_pca = pca_model.transform(filtered_set)
    inverse_training_pca = pca_model.inverse_transform(training_pca)
    residuals = filtered_set - inverse_training_pca
    residuals = abs(np.sum(residuals, axis=1))
    plt.plot(residuals)
    plt.title('Residuals on filtered training data')
    plt.show()

    # create residual graph with testing data
    test_pca = pca_model.transform(no_dates_test_set)
    inverse_test_pca = pca_model.inverse_transform(test_pca)
    residuals = no_dates_test_set - inverse_test_pca
    residuals = abs(np.sum(residuals, axis=1))
    plt.plot(residuals)
    plt.title('Residuals on test data')
    plt.show()

    for i in range(1, len(residuals)):
        if residuals[i] > 325:
            anomaly_set.iloc[i]['anomaly'] = 1
    return anomaly_set

answer = input("Do you want to perform the complete grid search for ARMA? (yes/ no): ")

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
test_set.set_index('date', inplace=True)
test_set['ATT_FLAG'] = 0
test_set.at['2017-01-16 09:59:59':'2017-01-19 06:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-01-30 08:59:59':'2017-02-02 00:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-02-09 03:59:59':'2017-02-10 09:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-02-12 01:59:59':'2017-02-13 07:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-02-24 05:59:59':'2017-02-28 08:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-03-10 09:59:59':'2017-03-13 21:59:59', 'ATT_FLAG'] = 1
test_set.at['2017-03-25 20:59:59':'2017-03-27 01:59:59', 'ATT_FLAG'] = 1

''' ARMA TASK '''
# Normalize the data and pick the 5 signals
train_data = training_set_1[['date', 'L_T1', 'L_T2', 'F_PU1', 'L_T3', 'L_T6']]
train_data = train_data.dropna()
train_data.set_index('date', inplace=True)

validation_data = training_set_2[['date', 'L_T1', 'L_T2', 'F_PU1', 'L_T3', 'L_T6']]
validation_data = validation_data.dropna()
validation_data.set_index('date', inplace=True)
validation_data['label'] = training_set_2['ATT_FLAG'].values

test_data = test_set[['L_T1', 'L_T2', 'F_PU1', 'L_T3', 'L_T6']]
test_data = test_set.dropna()
test_data['label'] = test_set['ATT_FLAG'].values

anomaly_set_arma = pd.DataFrame(index=test_data.index)
anomaly_set_arma['anomaly'] = 0
anomaly_set_discrete = pd.DataFrame(index=test_data.index)
anomaly_set_discrete['anomaly'] = 0
anomaly_set_pca = pd.DataFrame(index=test_data.index)
anomaly_set_pca['anomaly'] = 0

print('training set: ', len(train_data))
print('validation set: ', len(validation_data))
print('test data: ', len(test_data))

#Only use the comment below if you want to compute the (partial) autocorrelation for each signal
#plot_autocorrelation(train_data)

#format >> parameter_orders = {feature_name: [maximum_p, maximum_q, optimal_order, threshold]}
parameter_orders = {'L_T1': [11, 7, (2,0,3), 0.42], 'L_T2': [6, 16, (3,0,3), 0.7], 'F_PU1': [8, 2, (2,0,1), 2.75], 'L_T3': [3, 2, (3,0,2), 0.8], 'L_T6': [2, 4, (2,0,4), 2.4]}

for i in train_data.columns:
    # Evaluate whether the prediction fits training set 1 well enough and find optimal parameters p and q. Especially on T1 the data fits good
    train_data['actual'] = (train_data[i] - train_data[i].mean()) / train_data[i].std()
    validation_data['actual'] = (validation_data[i] - validation_data[i].mean()) / validation_data[i].std()
    test_data['actual'] = (test_data[i] - test_data[i].mean()) / test_data[i].std()

    if (answer == 'yes'):
        print('Finding optimal parameters p and q for ' , i, ' .......')
        model = find_optimal_parameters(train_data['actual'], parameter_orders[i][0], parameter_orders[i][1])
        action = model.order
    else: 
        action = parameter_orders[i][2]

    print('Building model on the first training set for ', i, ' .......')
    train_result = build_and_predict_model(train_data, action, parameter_orders[i][3])
    visualize_model(train_result, False, i)
    # Use training set 2 to optimize the model and to find the correct threshold
    print('Building model on the second training set for ', i, ' with attacks ......')
    validation_result = build_and_predict_model(validation_data, action, parameter_orders[i][3])
    visualize_model(validation_result, True, i)

    # Use test set to detect anomalies on new unseen data
    print('Building model on the test set for ', i, '........')
    test_result = build_and_predict_model(test_data, action, parameter_orders[i][3])
    visualize_model(test_result, True, i)
    anomaly_set_arma.loc[test_result['anomalies'] == 1] = 1

'''DISCRETE TASK'''
anomaly_set_discrete = discretization(test_data, train_data, anomaly_set_discrete, win_size=10, paa_size=6, alphabet_size=3, signal='L_T3')
anomaly_set_discrete = discretization(test_data, train_data, anomaly_set_discrete, win_size=16, paa_size=5, alphabet_size=3, signal='L_T2')
anomaly_set_discrete = discretization(test_data, train_data, anomaly_set_discrete, win_size=5, paa_size=5, alphabet_size=3, signal='F_PU1')
anomaly_set_discrete = discretization(test_data, train_data, anomaly_set_discrete, win_size=30, paa_size=5, alphabet_size=3, signal='L_T1')
anomaly_set_discrete = discretization(test_data, train_data, anomaly_set_discrete, win_size=20, paa_size=5, alphabet_size=3, signal='L_T6')

'''PCA TASK'''
anomaly_set_pca = pca(training_set_1, test_set, anomaly_set_pca)

print('Confusion matrix: ')
print('ARMA model')
print(confusion_matrix(test_data['label'], anomaly_set_arma['anomaly']))
print('Discrete model')
print(confusion_matrix(test_data['label'], anomaly_set_discrete['anomaly']))
print('PCA')
print(confusion_matrix(test_data['label'], anomaly_set_pca['anomaly']))