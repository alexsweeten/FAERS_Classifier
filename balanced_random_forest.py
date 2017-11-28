# Random Forest Algorithm on Sonar Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
import math

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def balanceSubsets(neg_set, pos_set, ratio):
    dataset_split = list()
    dataset_copy = list(neg_set)
    subset_size = math.floor(len(neg_set) / ratio)
    for i in range(int(ratio)):
        subset = list()
        for element in pos_set:
            subset.append(element)
        while len(subset) < 2 * subset_size + 1:
            index = randrange(len(dataset_copy))
            subset.append(dataset_copy.pop(index))
        dataset_split.append(subset)
    return(dataset_split)

def createSubset(dataset):
    dataset_split = list()
    negative_dataset = list()
    positive_dataset = list()
    for i in dataset:
        if (i[len(i)-1] == 0):
            negative_dataset.append(i)
        else:
            positive_dataset.append(i)
    return(negative_dataset, positive_dataset)

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

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

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, ratio, *args):
	#folds = cross_validation_split(dataset, n_folds)
	folds = balanceSubsets(x[0], x[1], ratio)
	print(len(folds))
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		sensitivity = sensitivity_metric(actual, predicted)
		ppv = ppv_metric(actual, predicted)
		fscore = fscore_metric(sensitivity, ppv)
		scores.append(accuracy)
		scores.append(sensitivity)
		scores.append(ppv)
		scores.append(fscore)
	return scores

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Select the best split point for a dataset
def get_split(dataset, n_features):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	features = list()
	while len(features) < n_features:
		index = randrange(len(dataset[0])-1)
		if index not in features:
			features.append(index)
	for index in features:
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, n_features, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left, n_features)
		split(node['left'], max_depth, min_size, n_features, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right, n_features)
		split(node['right'], max_depth, min_size, n_features, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size, n_features):
	root = get_split(train, n_features)
	split(root, max_depth, min_size, n_features, 1)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

def getRatio(dataset):
    num_negative = 0
    num_positive = 0
    for j in dataset:
        if (j[len(j)-1]) == 0:
            num_negative += 1
        else:
            num_positive += 1
    ratio = num_negative / float(num_positive)
    return math.ceil(ratio)

# Create a random subsample from the dataset with replacement
def subsample(dataset, ratio):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(dataset))
		sample.append(dataset[index])
	return sample

# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
	predictions = [predict(tree, row) for tree in trees]
	return max(set(predictions), key=predictions.count)

# Random Forest Algorithm
def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
	trees = list()
	for i in range(n_trees):
		sample = subsample(train, sample_size)
		tree = build_tree(sample, max_depth, min_size, n_features)
		trees.append(tree)
	predictions = [bagging_predict(trees, row) for row in test]
	return(predictions)

# load and prepare data
filename = 'master_table_new.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(0, len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
k = getRatio(dataset)

x = createSubset(dataset)
print('# of Negative Class:  ' +  str(len(x[0])))
print('# of Positive Class:  ' +  str(len(x[1])))
balanceSubsets(x[0], x[1], k)

# evaluate algorithm
n_folds = 5
max_depth = 10
min_size = 1
sample_size = 1.0
n_features = int(sqrt(len(dataset[0])-1))
for n_trees in [1, 5, 10]:
	scores = evaluate_algorithm(dataset, random_forest, k, max_depth, min_size, sample_size, n_trees, n_features)
	print('Trees: %d' % n_trees)
	print('Scores: %s' % scores)
