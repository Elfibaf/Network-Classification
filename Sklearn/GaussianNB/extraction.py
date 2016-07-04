import re
import numpy as np
import os
import matplotlib.pyplot as py
import arff
from sklearn.cross_validation import train_test_split,KFold

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

#Conversion of string attributes to float
def string_attributes_to_float(nparray):

    print nparray
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

#Creates a DataSet from an arff file
def load_dataset(filename):

    barray = []
    for row in arff.load(filename):
         barray.append(list(row))
    labels = []
    for row in barray:
        labels.append(row.pop())
        #row.pop()
    nparray=np.array(barray)
    labels = np.array(labels) 

    return DataSet(nparray,labels,0)

def split_data(filename):

    feature_train,feature_test,label_train,label_test = train_test_split(filename.features,filename.labels,test_size=0.25,random_state=42)

    feature_train_float = feature_train.astype(np.float)
    feature_test_float = feature_test.astype(np.float)

    return feature_train_float,feature_test_float,label_train,label_test

def kfold_data(filename,num_folds):
    
    Kf = KFold(filename.nb_examples,n_folds=num_folds,shuffle=True)
    for train_indices,test_indices in Kf:
        feature_train,feature_test = [filename.features[i] for i in train_indices],[filename.features[j] for j in test_indices]
        label_train,label_test = [ filename.labels[k] for k in train_indices],[filename.labels[l] for l in test_indices]
	
    feature_test_list = np.asarray(feature_test)
    feature_train_list = np.asarray(feature_train)
    
    return feature_train,feature_test,label_train,label_test

def plot_confusion_matrix(cm,clf,title="Confusion matrix",cmap=py.cm.Blues):
    
    py.imshow(cm,interpolation="nearest",cmap=cmap)
    py.title(title)
    py.colorbar()
    tick_marks = np.arange(len(clf.classes_))
    py.xticks(tick_marks,clf.classes_,rotation=45)
    py.yticks(tick_marks,clf.classes_)
    py.tight_layout()
    py.ylabel("True label")
    py.xlabel("Predicted label")
    

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

    features = features.T[1:].astype(np.float)
    labels = np.asarray(labels)

    return labels,features

def labels_into_1hot_vector(labels):

    dict_labels = {} #dictionnary containing index of each label in the 1-hot vector
    i = 0
    for label in labels:
        if not label in dict_labels:
            dict_labels[label] = i
            i += 1
    labels_one_hot = np.zeros((len(labels),len(dict_labels)), dtype = 'i')
    for i in range(len(labels)):
        labels_one_hot[i][dict_labels[labels[i]]] = 1

    return labels_one_hot

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
    
    # Extract labels and convert
    
    labels = []
    labels,features = extract_labels(labels,features)

    return DataSet(features,labels[1:],0)

