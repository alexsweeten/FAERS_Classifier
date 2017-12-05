from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from csv import reader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

def random_forest(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def decision_tree(features, target):
    clf = DecisionTreeClassifier()
    clf.fit(features, target)
    return clf

def svm(features, target):
    clf = SVC()
    clf.fit(features, target)
    return clf

def balance_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

def dataset_statistics(dataset):
    print dataset.describe()

def getRatio(dataset):
    num_negative = 0
    num_positive = 0
    for element in dataset:
        if (element) == 0:
            num_negative += 1
        else:
            num_positive += 1
    return(num_negative, num_positive)

#calculate sensitivity  (true positive rate)
def sensitivity_metric(actual, predicted):
    true_positives = 0
    false_negatives = 0
    for i in range(len(actual)):
        if (actual[i] == predicted[i]) and actual[i] == 1:
            true_positives += 1
        elif (actual[i] != predicted[i]) and actual[i] == 1:
            false_negatives += 1
    try:
        sensitivity_score = true_positives / (float(true_positives) + float(false_negatives))
        return sensitivity_score
    except:
        return 0

#calculate sensitivity  (true positive rate)
def ppv_metric(actual, predicted):
    true_positives = 0
    false_positives = 0
    for i in range(len(actual)):
        if (actual[i] == predicted[i]) and actual[i] == 1:
            true_positives += 1
        elif (actual[i] != predicted[i]) and actual[i] == 0:
            false_positives += 1
    try:
        ppv_score = true_positives / (float(true_positives) + float(false_positives))
        return ppv_score
    except:
        return 0

#calculate sensitivity  (true positive rate)
def fscore_metric(sensitivity, ppv):
    try:
        return (2 * sensitivity * ppv) / (float(sensitivity) + float(ppv))
    except:
        return 0

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

def main():
    counter = 1
    model_list = []

    for filename in os.listdir('./clipped_datasets'):
        #compute a random forest for each of the split datasets
        dataset = pd.read_csv("./clipped_datasets/" + filename)
        headers = list(dataset)
        train_x, test_x, train_y, test_y = split_dataset(dataset, 0.5, headers[1:-1], headers[-1])
        trained_model = random_forest(train_x, train_y)
        model_list.append(trained_model)

    trained_model = model_list[0]
    for i in range(1, len(model_list)):
        trained_model = balance_rfs(trained_model, model_list[i])


    filename = 'master_table_full.csv'
    dataset = pd.read_csv(filename)
    headers = list(dataset)
    last_column = dataset.iloc[:,-1]
    classes = getRatio(last_column)

    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.5, headers[1:-1], headers[-1])
    single_tree = decision_tree(train_x, train_y)
    random_tree = random_forest(train_x, train_y)
    support_vector = svm(train_x, train_y)
    predictions = trained_model.predict(test_x)
    predictions_2 = single_tree.predict(test_x)
    predictions_3 = random_tree.predict(test_x)
    predictions_4 = support_vector.predict(test_x)

    print('# Negative Class: %s' % classes[0])
    print('# Positive Class: %s' % classes[1])
    print('Ratio -/+: %s' % (classes[0]/float(classes[1])))
    print('\n')
    print('--- Balanced Random Forest Results ---')
    print('Accuracy: %s' % accuracy_score(test_y, predictions))
    sens = sensitivity_metric(list(test_y), predictions)
    print('Sensitivity: %s' % sens)
    ppv = ppv_metric(list(test_y), predictions)
    print('PPV: %s' % ppv)
    print('F-Score %s' % fscore_metric(sens, ppv))
    weights = trained_model.feature_importances_
    weight_dictionary = {}

    for i in range(0, len(weights)):
        weight_dictionary.update({headers[i]: weights[i]})
    for key, value in sorted(weight_dictionary.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)

    print('\n')
    print("--- Unbalanced Random Forest Results ---")

    print('Accuracy: %s' % accuracy_score(test_y, predictions_3))
    sens_3 = sensitivity_metric(list(test_y), predictions_3)
    print('Sensitivity: %s' % sens_3)
    ppv_3 = ppv_metric(list(test_y), predictions_3)
    print('PPV: %s' % ppv_3)
    print('F-Score %s' % fscore_metric(sens_3, ppv_3))

    print('\n')
    print("--- Single Decision Tree Results ---")

    print('Accuracy: %s' % accuracy_score(test_y, predictions_2))
    sens_2 = sensitivity_metric(list(test_y), predictions_2)
    print('Sensitivity: %s' % sens_2)
    ppv_2 = ppv_metric(list(test_y), predictions_2)
    print('PPV: %s' % ppv_2)
    print('F-Score %s' % fscore_metric(sens_2, ppv_2))

    plt.figure(1)
    plt.rcParams.update({'font.size': 12})
    plt.plot(fscore_metric(sens, ppv), sens, 'ro')
    plt.plot(fscore_metric(sens_3, ppv), sens_3, 'bs')
    plt.plot(fscore_metric(sens_2, ppv), sens_2, 'g^')

    plt.ylabel('Sensitivity')
    plt.title('Classifier Sensitivity over F-Score')
    plt.xlabel('F-Score')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
