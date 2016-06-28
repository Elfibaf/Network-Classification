import re
import numpy as np
import os
import arff

from sklearn.cross_validation import train_test_split,KFold

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

    for i in range(len(nparray[3])):
        nparray[1][i] = ''.join(nparray[1][i].split('.'))
        nparray[3][i] = ''.join(nparray[3][i].split('.'))

    nparray = nparray.T
    

    return DataSet(nparray,labels)


#Loads a dataset from a .info file from the Barcelona data
def load_dataset_barcelona(filename):

    #Opening the file and reading it
    info_file = open(filename)
    text = info_file.read()

    #regex to capture every line of the file into a list
    regular_expression = r"(.+#)+"
    res = re.findall(regular_expression, text)

    #for each line, we split the features according to the "#" separator.
    #We get a 2-dimensions (14*nb_flows) list
    res = [l.split("#") for l in res]

    #Getting rid of the unlabelled data + the last 3 features that are useless
    for flow in res:
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop()
        flow.pop(-2)
        

    #list of indexes of the unlabelled flow to delete them
    indexes = []
    for i in range(len(res)):
        if res[i][8] == '-':        
            indexes.append(i)
    res = np.asarray(res)
    
    #Deleting flows with the above list of indexes
    features = np.delete(res, indexes, 0).tolist()


    #Extracting labels
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
