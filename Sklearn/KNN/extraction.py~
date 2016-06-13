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


    #Conversion of string attributes to float
    nparray = nparray.T
    for i in range(len(nparray[3])):
        if nparray[3][i] == 'TCP':
            nparray[3][i] = 0
        elif nparray[3][i] == 'UDP':
            nparray[3][i] = 1

    for i in range(len(nparray[4])):
        nparray[4][i] = ''.join(nparray[4][i].split('.'))
        nparray[6][i] = ''.join(nparray[6][i].split('.'))

    nparray = nparray.T
    

    return DataSet(nparray,labels)


#foo = load_dataset('../Data/Capture_Port.arff')
#print(foo.features, foo.labels, foo.nb_examples)



