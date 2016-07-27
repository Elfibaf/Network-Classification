import re
import numpy as np
import os
import pickle
import matplotlib.pyplot as py
import arff

class DataSet(object):
    def __init__(self, features, labels, l):
        self._features = features 
        self._labels = labels 
        self._nb_examples = labels.shape[0]
        self._l = l

    @property
    def features(self):
        return self._features

    @property
    def labels(self):
        return self._labels
        
    @property
    def l(self):
        return self._l

    @property
    def nb_examples(self):
        return self._nb_examples
        
    @property
    def nb_classes(self):
        return len({key: None for key in self._l})

def extract_labels(labels,features):

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

    features = features.T[1:]#.astype(np.float)
    features = [" ".join(feature) for feature in features]
    labels.pop(0)
    labels = np.asarray(labels)

    return labels,features

def labels_into_1hot_vector(labels,train):

    if train:
    	dict_labels = {} #dictionnary containing index of each label in the 1-hot vector
    	i = 0
    	for label in labels:
       	 if not label in dict_labels:
           	 dict_labels[label] = i
            	 i += 1
    	save_obj(dict_labels, "dict_labels")

    else:
        dict_labels = load_obj("dict_labels")

    labels_one_hot = np.zeros((len(labels),len(dict_labels)), dtype = 'i')
    for i in range(len(labels)):
        labels_one_hot[i][dict_labels[labels[i]]] = 1

    return labels_one_hot,dict_labels

def load_dataset_info(filename,train=True):

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
    
    # Extract labels, store them in a pickle and convert them
    
    labels = []
    labels,features = extract_labels(labels,features)
    labels_one_hot,dict_labels = labels_into_1hot_vector(labels,train)
    
    return DataSet(features, labels_one_hot,dict_labels)

#To save an object into a file
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# To load an object 
def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

