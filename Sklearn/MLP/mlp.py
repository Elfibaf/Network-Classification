# Authors : Mehdi Crozes and Fabien Robin
# Date : June 9th 2016

# MLP for ARFF traffic network file 

import arff
import numpy as np
import time

from extraction_feature_selection import *
from sklearn.neural_network import MLPClassifier 
from sklearn.cross_validation import train_test_split,KFold
from sklearn.metrics import confusion_matrix,recall_score,precision_score

# Step 1 : Import Arff file

#arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")
arff_file = load_dataset_barcelona("../../Data/Info_file/packets_all_1.info")
print "\tTotal dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set

#feature_train,feature_test,label_train,label_test = split_data(arff_file)

feature_train,feature_test,label_train,label_test = kfold_data(arff_file,10)

# Step 3 : Standardization of feature_train and feature_test : theirs values will be now between 0. and 1.

scaler = MinMaxScaler(copy='false')
feature_train_rescaled = scaler.fit_transform(feature_train)
feature_test_rescaled = scaler.fit_transform(feature_test) 


# Step 4 : Fitting and training our classifier 
# Before that, we had to convert our feature_train_float into float type because our feature_train is an array of string and generate a TypeError

t0 = time.time()
clf = MLPClassifier(hidden_layer_sizes=(3,500), max_iter=20, alpha=1e-5,
                    activation='relu',algorithm='sgd', verbose='true', tol=1e-4, random_state=1,
                    learning_rate_init=.1)
clf.fit(feature_train_rescaled,label_train)
print "\tTraining  time ",round(time.time()-t0,3),"s"

t1= time.time()
label_pred=clf.predict(feature_test_rescaled)
print "\tPredicting time ",round(time.time()-t1,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

# Step 5 :  Precision and recall score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# Step 6 : Accuracy 

score = clf.score(feature_test_rescaled,label_test)
score2 = clf.score(feature_train_rescaled,label_train)
print "\tTest Accuracy :",score 
print "\tTraining Accuracy :",score2


