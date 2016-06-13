# Authors : Mehdi Crozes and Fabien Robin
# Date : June 8th 2016

# Perceptron classifier for ARFF traffic network file 

import arff
import numpy as np
import time

from extraction import *
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,precision_score,recall_score
from sklearn.preprocessing import MinMaxScaler

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")
print "Total dataset : "
print "Number of features :",len(arff_file.features)
print "Number of labels :",len(arff_file.labels)

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Fitting and training our classifier 
# Before that, we had to convert our feature_train_float into float type because our feature_train is an array of string and generate a TypeError

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

scaler = MinMaxScaler(copy='false')
feature_train_rescaled =scaler.fit_transform(feature_train_float)
feature_test_rescaled=scaler.fit_transform(feature_test_float) 

print feature_train_rescaled,feature_test_rescaled

clf = Perceptron(n_iter=50)
t0 = time.time()
clf.fit(feature_train_float,label_train)
print "\tTraining time ",round(time.time()-t0,3),"s"

t1= time.time()
label_pred = clf.predict(feature_test_float)
print "\tPredicting time ",round(time.time()-t1,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

# Step 4 : Precision and recall score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# Step 5 : Accuracy 

score = clf.score(feature_test_rescaled,label_test)
score2 = clf.score(feature_train_rescaled,label_train)
print "\tTest Accuracy :",score 
print "\tTraining Accuracy :",score2


