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
    """Dataset class for features,labels and number of examples from file"""

    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._nb_examples = labels.shape[0]

    @property
    def features(self):
        """Return the features """
        return self._features

    @property
    def labels(self):
        """Return  the labels """
        return self._labels

    @property
    def nb_examples(self):
        """Return the number of examples contained in the dataset"""
        return self._nb_examples

def load_dataset(filename):

    """Creates a DataSet from an arff file"""
    barray = []
    for row in arff.load(filename):
        barray.append(list(row))
    labels = []
    for row in barray:
        labels.append(row.pop())
        #row.pop()
    nparray = np.array(barray)
    labels = np.array(labels)


    """Conversion of string attributes to float
    nparray = nparray.T
    for i in range(len(nparray[3])):
        if nparray[3][i] == 'TCP':
            nparray[3][i] = 0
        elif nparray[3][i] == 'UDP':
            nparray[3][i] = 1

    for i in range(len(nparray[6])):
        nparray[4][i] = ''.join(nparray[4][i].split('.'))
        nparray[6][i] = ''.join(nparray[6][i].split('.'))

    nparray = nparray.T
    """
    return DataSet(nparray, labels)

def load_dataset_barcelona(filename):
    """Loads a dataset from a .info file from the Barcelona data"""

    info_file = open(filename)
    text = info_file.read()

    regular_expression = r"(.+#)+"
    list_lines = re.findall(regular_expression, text)

    """We get a 2-dimensions (14*nb_flows) list"""
    list_lines_splited = [l.split("#") for l in list_lines]

    """ Get unlabelled data + the last useless 3 features"""
    for flow in list_lines_splited:
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop(-2)

    """list of indexes of the unlabelled flow to delete them"""
    indexes = []
    for i in range(0, len(list_lines_splited)):
        if list_lines_splited[i][8] == '-':
            indexes.append(i)
    list_lines_splited_array = np.asarray(list_lines_splited)

    """Deleting flows with the above list of indexes"""
    features = np.delete(list_lines_splited_array, indexes, 0).tolist()

    labels = []
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

    return DataSet(features, labels[1:])

def split_data(arff_file):
    """Split implementation and convert features into float array"""

    feature_train, feature_test, label_train, label_test = train_test_split(arff_file.features, arff_file.labels, test_size=0.25, random_state=42)

    feature_train_float = feature_train.astype(np.float)
    feature_test_float = feature_test.astype(np.float)

    return feature_train_float, feature_test_float, label_train, label_test

def kfold_data(arff_file, num_folds):
    """KFold implementation to build training set and testing set"""

    kfold_list = KFold(arff_file.nb_examples, n_folds=num_folds, shuffle=True)
    for train_indices, test_indices in kfold_list:
        feature_train, feature_test = [arff_file.features[i] for i in train_indices], [arff_file.features[j] for j in test_indices]
        label_train, label_test = [arff_file.labels[k] for k in train_indices], [arff_file.labels[l] for l in test_indices]

    feature_test_list = np.asarray(feature_test)
    feature_train_list = np.asarray(feature_train)

    return feature_train_list, feature_test_list, label_train, label_test

def plot_confusion_matrix(cm, clf, title="Confusion matrix", cmap=py.cm.Blues):
    """Plotting matrix_confusion for information"""

    py.imshow(cm, interpolation="nearest", cmap=cmap)
    py.title(title)
    py.colorbar()
    tick_marks = np.arange(len(clf.classes_))
    py.xticks(tick_marks, clf.classes_, rotation=45)
    py.yticks(tick_marks, clf.classes_)
    py.tight_layout()
    py.ylabel("True label")
    py.xlabel("Predicted label")
