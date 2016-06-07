import re
import numpy as np
import os
import arff



class DataSet(object):
    def __init__(self, features, labels):
        self._features = features 
        self._labels = labels 
        self._num_examples = labels.shape[0]

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples


#Creates a DataSet from an arff file
def load_dataset(filename):
    barray = []
    for row in arff.load(filename):
         barray.append(list(row))
    labels = []
    for row in barray:
        labels.append(row.pop())
    nparray = np.array(barray)
    labels = np.array(labels)

    return DataSet(nparray,labels)


#foo = load_dataset('../Data/Capture_Port.arff')
#print(foo.features, foo.labels, foo.num_examples)

