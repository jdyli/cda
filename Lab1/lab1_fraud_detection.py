"""
Cyber Data Analytics assignment lab 1

For each part of the assignment there are separate functions. Each subsection of the code
that corresponds to the part of the assignment is indicated with capital letters.

Run in python 3. If the (certain) libraries below are not yet installed, you can use the following pip-commands:

pip install matplotlib
pip install numpy
pip install pandas
pip install seaborn
pip sklearn
pip install imbalanced-learn
pip install graphviz

"""

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
import graphviz

pd.options.mode.chained_assignment = None  # default='warn'
warnings.simplefilter(action='ignore', category=FutureWarning)

"""EXPLORING DATA"""
def exploring(data):
    print(data.info())
    print(data.head())
    print('Empty values: ', data.isnull().values.sum())
    print('Chargeback:', len(data.loc[data['simple_journal'] == 'Chargeback']))
    print('Refused: ', len(data.loc[data['simple_journal'] == 'Refused']))
    print('Authorised: ', len(data.loc[data['simple_journal'] == 'Authorised']))
    print('Settled: ', len(data.loc[data['simple_journal'] == 'Settled']))
    # Exploring non-integer features
    print('Unique countries: ', data.issuercountrycode.unique(), ' total: ', len(data.issuercountrycode.unique()))
    print('txvariantcode: ', data.txvariantcode.unique(), ' total: ', len(data.txvariantcode.unique()))
    print('currencycode: ', data.currencycode.unique(), ' total: ', len(data.currencycode.unique()))       
    print('shoppercountrycode: ', data.shoppercountrycode.unique(), ' total: ', len(data.shoppercountrycode.unique()))         
    print('shopperinteraction: ', data.shopperinteraction.unique(), ' total: ', len(data.shopperinteraction.unique()))        
    print('cardverificationcodesupplied: ', data.cardverificationcodesupplied.unique(), ' total: ', len(data.cardverificationcodesupplied.unique()))         
    print('accountcode: ', data.accountcode.unique(), ' total: ', len(data.accountcode.unique()))                    
    print('mail_id: ', data.mail_id.unique(), ' total: ', len(data.mail_id.unique()))                
    print('ip_id: ', data.ip_id.unique(), ' total: ', len(data.ip_id.unique()))                  
    print('card_id: ', data.card_id.unique(), ' total: ', len(data.card_id.unique()))

def convert_categorical_to_numeric(data):
    data.issuercountrycode, mapping_issuercountrycode = pd.Series(data.issuercountrycode).factorize()
    data.txvariantcode, mapping_txvariantcode = pd.Series(data.txvariantcode).factorize()
    data.currencycode, mapping_currencycode = pd.Series(data.currencycode).factorize()
    data.shoppercountrycode, mapping_shoppercountrycode = pd.Series(data.shoppercountrycode).factorize()
    data.shopperinteraction, mapping_shopperinteraction = pd.Series(data.shopperinteraction).factorize()
    data.cardverificationcodesupplied, mapping_cardverificationcodesupplied = pd.Series(data.cardverificationcodesupplied).factorize()
    data.accountcode, mapping_accountcode = pd.Series(data.accountcode).factorize()
    data.mail_id, mapping_mail_id = pd.Series(data.mail_id).factorize()
    data.ip_id, mapping_ip_id = pd.Series(data.ip_id).factorize()
    data.card_id, mapping_card_id = pd.Series(data.card_id).factorize()

    mapping = {'issuercountrycode': mapping_issuercountrycode, 'txvariantcode': mapping_txvariantcode,
                'currencycode': mapping_currencycode, 'shoppercountrycode': mapping_shoppercountrycode,
                'shopperinteraction': mapping_shopperinteraction, 'cardverificationcodesupplied': mapping_cardverificationcodesupplied,
                'accountcode': mapping_accountcode, 'mail_id': mapping_mail_id,
                'ip_id': mapping_ip_id, 'card_id': mapping_card_id}
    return data, mapping

def preprocessing(data):
    data = shuffle(data)
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data = data.drop('txid', 1)
    data = data.drop('bookingdate', 1)
    data['creationdate'] =  pd.to_datetime(data['creationdate'], format='%Y-%m-%d %H:%M:%S')
    data['creationdate'] = pd.to_numeric(data['creationdate'])
    (data, mapping) = convert_categorical_to_numeric(data)
    data = data.loc[data['simple_journal'] != 'Refused']
    data['simple_journal'] = data['simple_journal'].replace(['Chargeback'], 1)
    data['simple_journal'] = data['simple_journal'].replace(['Settled', 'Authorised'], 0)
    data['cvcresponsecode'] = data['cvcresponsecode'].replace([3, 4, 5, 6], 3)
    #Taking currency level at time of writing; converted in Euro
    currencies = {'MXN': 0.04707, 'AUD': 0.62736, 'NZD': 0.59211, 'GBP': 1.16351, 'SEK': 0.09372} 
    data['amount'].loc[data['currencycode'] == 'MXN'] = data['amount'].loc[data['currencycode'] == 'MXN'] * currencies['MXN']
    data['amount'].loc[data['currencycode'] == 'AUD'] = data['amount'].loc[data['currencycode'] == 'AUD'] * currencies['AUD']
    data['amount'].loc[data['currencycode'] == 'NZD'] = data['amount'].loc[data['currencycode'] == 'NZD'] * currencies['NZD']
    data['amount'].loc[data['currencycode'] == 'GBP'] = data['amount'].loc[data['currencycode'] == 'GBP'] * currencies['GBP']
    data['amount'].loc[data['currencycode'] == 'SEK'] = data['amount'].loc[data['currencycode'] == 'SEK'] * currencies['SEK']
    return data, mapping

""" (1) VISUALIZATION TASK"""
def boxplot_amount(data):
    sns.set()
    sns.boxplot(data=data, x='simple_journal', y='amount')
    plt.show()
    plt.close()

def histogram_currencycode(total_non, total_fraudulent, mapping):
    number_of_transactions_non = []
    number_of_transactions_fraudulent = []

    for currency in range(0, len(mapping['currencycode'])):
        transactions_chargeback = (len(total_fraudulent.loc[(total_fraudulent['currencycode'] == currency)]) / len(total_fraudulent)) * 100
        number_of_transactions_fraudulent.append(transactions_chargeback)
        transactions_settled = (len(total_non.loc[(total_non['currencycode'] == currency)]) / len(total_non)) * 100
        number_of_transactions_non.append(transactions_settled)

    x = np.arange(len(mapping['currencycode']))
    plt.bar(x, height= number_of_transactions_fraudulent)
    plt.bar(x, height=number_of_transactions_non) 
    plt.ylabel('Currency type of transactions in %')
    plt.xticks(x, mapping['currencycode'])
    plt.legend(('Fraudulent', 'Benign'))
    plt.show()
    plt.close()

def piechart_txvariantcode(total_non, total_fraudulent, mapping):
    txvariantcode = mapping['txvariantcode']
    number_of_transactions_non = []
    number_of_transactions_fraudulent = []

    for code in range(0, len(txvariantcode)):
        transactions_txvariantcode = (len(total_fraudulent.loc[total_fraudulent['txvariantcode'] == code]) / len(total_fraudulent)) * 100
        number_of_transactions_fraudulent.append(transactions_txvariantcode)
        transactions_txvariantcode_non = (len(total_non.loc[total_non['txvariantcode'] == code]) / len(total_non)) * 100
        number_of_transactions_non.append(transactions_txvariantcode_non)       

    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(txvariantcode, number_of_transactions_fraudulent)]
    patches, texts = plt.pie(number_of_transactions_fraudulent, startangle=90, radius=1.2)
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.))
    plt.title('Fraudulent')
    plt.show()
    plt.close()

    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(txvariantcode, number_of_transactions_non)]
    patches, texts = plt.pie(number_of_transactions_non, startangle=90, radius=1.2)
    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.))
    plt.title('Benign')
    plt.show()
    plt.close()

def correlationmatrix(data):
    new_data = data[['amount', 'cardverificationcodesupplied', 'cvcresponsecode', 'simple_journal']]
    sns.heatmap(new_data.corr(), xticklabels=new_data.corr().columns, yticklabels=new_data.corr().columns)
    plt.show()
    plt.close()

def show_visualizations(data, mapping):
    total_non = data.loc[data['simple_journal'] == 0]
    total_fraudulent = data.loc[data['simple_journal'] == 1]
    histogram_currencycode(total_non, total_fraudulent, mapping)
    piechart_txvariantcode(total_non, total_fraudulent, mapping)
    correlationmatrix(data)
    boxplot_amount(data.sample(frac=0.1))

""" (2) IMBALANCE TASK"""
def fraud_detection(data):
    data = data.drop('cardverificationcodesupplied', 1)
    data = data.drop('cvcresponsecode', 1)
    y = data['simple_journal']
    X = data.drop('simple_journal', 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35) 
    sm = SMOTE(ratio = 1)
    x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train)

    print("\nBefore SMOTE, counts of label '1': {}".format(sum(y_train==1)))
    print("Before SMOTE, counts of label '0': {} \n".format(sum(y_train==0)))
    print("\nAfter SMOTE, counts of label '1': {}".format(sum(y_train_sm==1)))
    print("After SMOTE, counts of label '0': {} \n".format(sum(y_train_sm==0)))
    print("\nTest data, counts of label '1': {}".format(sum(y_test==1)))
    print("Test data, counts of label '0': {} \n".format(sum(y_test==0)))

    # (1) Build Random Forest classifier (Black Box)
    # with SMOTE
    clf_rf_sm = RandomForestClassifier(n_estimators=25, random_state=12)
    clf_rf_sm.fit(x_train_sm, y_train_sm)

    probs_rf_sm = clf_rf_sm.predict_proba(x_test)
    probs_rf_sm = probs_rf_sm[:,1]
    auc_rf_sm = roc_auc_score(y_test, probs_rf_sm)
    fpr_rf_sm, tpr_rf_sm, thresholds_rf_sm = roc_curve(y_test, probs_rf_sm)

    # without SMOTE
    clf_rf = RandomForestClassifier(n_estimators=25, random_state=12)
    clf_rf.fit(x_train, y_train)

    probs_rf = clf_rf.predict_proba(x_test)
    probs_rf = probs_rf[:,1]
    auc_rf = roc_auc_score(y_test, probs_rf)
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, probs_rf)

    # (2) Build KNN Classifier (Black Box)
    # with SMOTE
    neigh_sm = KNeighborsClassifier(n_neighbors=3)
    neigh_sm.fit(x_train_sm, y_train_sm)

    probs_neigh_sm = neigh_sm.predict_proba(x_test)
    probs_neigh_sm = probs_neigh_sm[:,1]
    auc_neigh_sm = roc_auc_score(y_test, probs_neigh_sm)
    fpr_neigh_sm, tpr_neigh_sm, thresholds_neigh_sm = roc_curve(y_test, probs_neigh_sm)

    # without SMOTE
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)

    probs_neigh = neigh.predict_proba(x_test)
    probs_neigh = probs_neigh[:,1]
    auc_neigh = roc_auc_score(y_test, probs_neigh)
    fpr_neigh, tpr_neigh, thresholds_neigh = roc_curve(y_test, probs_neigh)

    # (3) Build Decision Tree Classifier (White Box)
    # with SMOTE
    dt_sm = tree.DecisionTreeClassifier()
    dt_sm.fit(x_train_sm, y_train_sm)

    probs_dt_sm = dt_sm.predict_proba(x_test)
    probs_dt_sm = probs_dt_sm[:,1]
    auc_dt_sm = roc_auc_score(y_test, probs_dt_sm)
    fpr_dt_sm, tpr_dt_sm, thresholds_dt_sm = roc_curve(y_test, probs_dt_sm)

    # without SMOTE
    dt = tree.DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    probs_dt = dt.predict_proba(x_test)
    probs_dt = probs_dt[:,1]
    auc_dt = roc_auc_score(y_test, probs_dt)
    fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, probs_dt)

    # Plot ROC curves
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(fpr_rf_sm, tpr_rf_sm, marker='.', label='Random Forest SMOTE (area= %0.2f)' % auc_rf_sm)
    plt.plot(fpr_rf, tpr_rf, marker='.', label='Random Forest (area= %0.2f)' % auc_rf)
    plt.plot(fpr_neigh_sm, tpr_neigh_sm, marker='.', label='kN SMOTE (area= %0.2f)' % auc_neigh_sm)
    plt.plot(fpr_neigh, tpr_neigh, marker='.', label='kN (area= %0.2f)' % auc_neigh)
    plt.plot(fpr_dt_sm, tpr_dt_sm, marker='.', label='Decision Tree SMOTE (area= %0.2f)' % auc_dt_sm)
    plt.plot(fpr_dt, tpr_dt, marker='.', label='Decision Tree (area= %0.2f)' % auc_dt)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.show()

""" (3) CLASSIFICATION TASK"""
def final_classifier(data):
    classifiers = ['decision_tree', 'random_forest']
    y = data['simple_journal'].as_matrix()
    X = data.drop('simple_journal', 1).as_matrix()

    train_features, test_features, train_target, test_target = train_test_split(X, y, test_size = 0.35)
    for i in classifiers:
        list_precision = []
        list_recall = []
        kf = StratifiedKFold(n_splits=10)
        for fold, (train_index, test_index) in enumerate(kf.split(train_features, train_target)):
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            sm = SMOTE(ratio = 1)
            X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)
            if (i == 'random_forest'):
                model = RandomForestClassifier(n_estimators=10, random_state=10)
            else:
                model = tree.DecisionTreeClassifier()
            model.fit(X_train_oversampled, y_train_oversampled)
            y_pred = model.predict(X_test)
            print(f'\n------------------------------------------------- \nFor fold {fold}:')
            y_pred2 = model.predict(test_features)
            list_precision.append(precision_score(test_target, y_pred2))
            list_recall.append(recall_score(test_target, y_pred2))
            print(f'Validation set - Confusion Matrix:\n {confusion_matrix(y_test, y_pred)}')
            print(f'Test set - Confusion Matrix:\n {confusion_matrix(test_target, y_pred2)}')
            print(f'Accuracy: {accuracy_score(test_target, y_pred2)}')
            print(f'Precision: {precision_score(test_target, y_pred2)}')
            print(f'Recall: {recall_score(test_target, y_pred2)}')
            if (fold == 9 and i == 'decision_tree'):
                # Visualize decision tree
                tree.export_graphviz(model, out_file='decision_tree.dot', max_depth = 5)
        average_precision = round(sum(list_precision) / len(list_precision), 2)
        average_recall = round(sum(list_recall) / len(list_recall), 2)
        print('Average precision: ', average_precision)
        print('Average recall: ', average_recall)

if __name__ == "__main__":
    print('Reading data..')
    data = pd.read_csv('data_for_student_case.csv')
    #exploring(data)
    print('Preprocessing data..')
    (preprocessed_data, mapping) = preprocessing(data)
    show_visualizations(preprocessed_data, mapping) # VISUALIZATION TASK
    print('Running classifier..') 
    fraud_detection(preprocessed_data) # IMBALANCE TASK
    final_classifier(preprocessed_data) # CLASSIFIER TASK