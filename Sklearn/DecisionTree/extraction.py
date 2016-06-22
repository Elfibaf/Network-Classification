import re
import numpy as np
import os
import arff

class DataSet(object):
    def __init__(self, features, labels):
        self._features = features 
        self._labels = labels 
        self._nb_examples = labels.shape[0]

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def nb_examples(self):
        return self._nb_examples


#Creates a DataSet from an arff file
def load_dataset(filename):
    barray = []
    for row in arff.load(filename):
         barray.append(list(row))
    labels = []
    for row in barray:
        labels.append(row.pop())
        #row.pop()
    nparray = np.array(barray)
    labels = np.array(labels)


    """#Conversion of string attributes to float
    nparray = nparray.T
    for i in range(len(nparray[3])):
        if nparray[3][i] == 'TCP':
            nparray[3][i] = 0
        elif nparray[3][i] == 'UDP':
            nparray[3][i] = 1

    for i in range(len(nparray[4])):
        nparray[4][i] = ''.join(nparray[4][i].split('.'))
        nparray[6][i] = ''.join(nparray[6][i].split('.'))

    nparray = nparray.T"""
    

    return DataSet(nparray,labels)

#Split implementation and convert features into float array

def split_data(arff_file):

    feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

    feature_train_float = feature_train.astype(np.float)
    feature_test_float = feature_test.astype(np.float)

    return feature_train_float,feature_test_float,label_train,label_test

# KFold implementation to build training set and testing set

def kfold_data(arff_file,num_folds):
    
    Kf = KFold(arff_file.nb_examples,n_folds=num_folds)
    for train_indices,test_indices in Kf:
        feature_train,feature_test = [arff_file.features[i] for i in train_indices],[arff_file.features[j] for j in test_indices]
        label_train,label_test = [ arff_file.labels[k] for k in train_indices],[arff_file.labels[l] for l in test_indices]

    feature_test_list = np.asarray(feature_test)
    feature_train_list = np.asarray(feature_train)

    return feature_train,feature_test,label_train,label_test
