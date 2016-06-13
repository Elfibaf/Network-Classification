# Authors : Mehdi Crozes and Fabien Robin
# Date : June 9th 2016

# MLP for ARFF traffic network file 

import arff
import numpy as np
import time

from extraction import *
from sklearn.neural_network import MLPClassifier 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_score

# Step 1 : Import Arff file

arff_file = load_dataset("../../../Data/data_caida_original.arff")
print "\tTotal dataset : "
print "\tNumber of features :",len(arff_file.features)
print "\tNumber of labels :",len(arff_file.labels)

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Fitting and training our classifier 
# Before that, we had to convert our feature_train_float into float type because our feature_train is an array of string and generate a TypeError

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

# Step 3bis : Standardization of feature_train and feature_test : theirs values will be now between 0. and 1.

scaler = MinMaxScaler(copy='false')
feature_train_rescaled =scaler.fit_transform(feature_train_float)
feature_test_rescaled=scaler.fit_transform(feature_test_float) 

t0 = time.time()
clf = MLPClassifier(hidden_layer_sizes=(3,500), max_iter=25, alpha=1e-5,
                    activation='relu',algorithm='sgd', verbose='true', tol=1e-4, random_state=1,
                    learning_rate_init=.1)
clf.fit(feature_train_rescaled,label_train)
print "\tTraining  time ",round(time.time()-t0,3),"s"

t1= time.time()
label_pred=clf.predict(feature_test_rescaled)
print "\tPredicting time ",round(time.time()-t1,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

# Step 4 :  Precision and recall score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# Step 5 : Accuracy 

score = clf.score(feature_test_rescaled,label_test)
score2 = clf.score(feature_train_rescaled,label_train)
print "\tTest Accuracy :",score 
print "\tTraining Accuracy :",score2


