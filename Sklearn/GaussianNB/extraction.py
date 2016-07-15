"""
Authors: Mehdi Crozes and Fabien Robin
Date : June 9th 2016
Update : June 28th 2016
"""

import re
import matplotlib.pyplot as py
import numpy as np
import arff
from sklearn.cross_validation import train_test_split, KFold

class DataSet(object):
    """Dataset class created from file"""
    def __init__(self, features, labels, l):
        self._features = features
        self._labels = labels
        self._nb_examples = labels.shape[0]
        self._l = l

    @property
    def features(self):
        """Return the features """
        return self._features

    @property
    def labels(self):
        """Return the labels """
        return self._labels

    @property
    def l(self):
        """Return the labels' dictionnary"""
        return self._l

    @property
    def nb_examples(self):
        """Return the number of examples"""
        return self._nb_examples

    @property
    def nb_classes(self):
        """Return the number of classes"""
        return len({key: None for key in self._l})


def string_attributes_to_float(nparray):
    """ Conversion of string attributes to float """
    nparray = nparray.T
    for i in range(len(nparray[3])):
        if nparray[3][i] == 'TCP':
            nparray[3][i] = 0
        elif nparray[3][i] == 'UDP':
            nparray[3][i] = 1

    for i in range(len(nparray[3])):
        nparray[1][i] = ''.join(nparray[1][i].split('.'))
        nparray[3][i] = ''.join(nparray[3][i].split('.'))

    nparray = nparray.T
    return nparray


def load_dataset(filename):
    """ Create a DataSet object from an arff file"""
    barray = []
    for row in arff.load(filename):
        barray.append(list(row))
    labels = []
    for row in barray:
        labels.append(row.pop())
        #row.pop()
    nparray = np.array(barray)
    labels = np.array(labels)

    return DataSet(nparray, labels, 0)


def split_data(filename):
    """Split implementation and convert features into float array"""

    feature_train, feature_test, label_train, label_test = train_test_split(filename.features, filename.labels, test_size=0.25, random_state=42)

    feature_train_float = feature_train.astype(np.float)
    feature_test_float = feature_test.astype(np.float)

    return feature_train_float, feature_test_float, label_train, label_test

def kfold_data(filename, num_folds):
    """KFold implementation to build training set and testing set"""

    kfold_list = KFold(filename.nb_examples, n_folds=num_folds, shuffle=True)
    for train_indices, test_indices in kfold_list:
        feature_train, feature_test = [filename.features[i] for i in train_indices], [filename.features[j] for j in test_indices]
        label_train, label_test = [filename.labels[k] for k in train_indices], [filename.labels[l] for l in test_indices]

    feature_test_list = np.asarray(feature_test)
    feature_train_list = np.asarray(feature_train)

    return feature_train_list, feature_test_list, label_train, label_test

def plot_confusion_matrix(cm, clf, title="Confusion matrix", cmap=py.cm.Blues):
    """ Plotting confusion_matrix for information"""

    py.imshow(cm, interpolation="nearest", cmap=cmap)
    py.title(title)
    py.colorbar()
    tick_marks = np.arange(len(clf.classes_))
    py.xticks(tick_marks, clf.classes_, rotation=45)
    py.yticks(tick_marks, clf.classes_)
    py.tight_layout()
    py.ylabel("True label")
    py.xlabel("Predicted label")


def extract_labels(labels, features):
    """ Extracting labels"""

    for flow in features:
        labels.append(flow.pop())
    features = np.asarray(features).T
    for i in range(len(features[7])):
        if features[7][i] == 'TCP':
            features[7][i] = 0
        elif features[7][i] == 'UDP':
            features[7][i] = 1

    for i in range(len(features[3])):
        features[3][i] = ''.join(features[3][i].split('.'))
        features[4][i] = ''.join(features[4][i].split('.'))

    features = features.T[1:].astype(np.float)
    labels = np.asarray(labels)

    return labels, features

def labels_into_1hot_vector(labels):
    """ Converting labels into 1hot vector for CNN"""

    """dictionnary containing index of each label in the 1-hot vector"""
    dict_labels = {}
    i = 0
    for label in labels:
        if not label in dict_labels:
            dict_labels[label] = i
            i += 1
    labels_one_hot = np.zeros((len(labels), len(dict_labels)), dtype='i')
    for i in range(len(labels)):
        labels_one_hot[i][dict_labels[labels[i]]] = 1

    return labels_one_hot

def load_dataset_barcelona(filename):
    """Loads a dataset from a .info file from the Barcelona data"""

    info_file = open(filename)
    text = info_file.read()

    #regex to capture every line of the file into a list
    regular_expression = r"(.+#)+"
    list_lines = re.findall(regular_expression, text)

    """We get a 2-dimensions (14*nb_flows) list"""
    list_lines_splitted = [l.split("#") for l in list_lines]

    #Getting rid of the unlabelled data + the last 3 features that are useless
    for flow in list_lines_splitted:
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop(-2)

    #list of indexes of the unlabelled flow to delete them
    indexes = []
    for i in range(len(list_lines_splitted)):
        if list_lines_splitted[i][8] == '-':
            indexes.append(i)
    list_lines_splitted_array = np.asarray(list_lines_splitted)

    #Deleting flows with the above list of indexes

    features = np.delete(list_lines_splitted_array, indexes, 0).tolist()

    labels = []
    labels, features = extract_labels(labels, features)

    return DataSet(features, labels[1:], 0)
