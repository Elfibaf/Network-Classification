import re
import numpy as np
import os
import arff
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split

class DataSet(object):

    def __init__(self, features, labels):
        self._features = features
        self._labels = labels
        self._nb_examples = labels.shape[0] 
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels

    @property
    def nb_examples(self):
        return self._nb_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def nb_classes(self):
        return len({key: None for key in self._labels})

    def next_batch(self, batch_size):
        
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._nb_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._nb_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._nb_examples
        end = self._index_in_epoch
        return self._features[start:end], self._labels[start:end]

    def normalize(self):
        scaler = MinMaxScaler(copy='false')
        self._features = scaler.fit_transform(self._features)

    def split(self):
        feature_train,feature_test,label_train,label_test = train_test_split(self._features,self._labels,test_size=0.25,random_state=42)
        train_data = DataSet(feature_train, label_train)
        test_data = DataSet(feature_test, label_test)
        return train_data, test_data


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


