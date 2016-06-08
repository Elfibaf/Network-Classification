# Authors : Mehdi Crozes and Fabien Robin
# Date : June 8th 2016

# Perceptron via SGDClassifier for ARFF traffic network file 

import arff
import numpy as np
import matplotlib.pyplot as py
import time

from extraction import *
from sklearn.linear_model import Perceptron
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")
print "Total dataset : "
print "Number of features :",len(arff_file.features)
print "Number of labels :",len(arff_file.labels)

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Modeling and training our classifier with training's time
# Before that, we had to convert our feature_train_float into float type because our feature_train is an array of string and generate a TypeError

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

scaler = MinMaxScaler(copy='false')
feature_train_rescaled =scaler.fit_transform(feature_train_float)
feature_test_rescaled=scaler.fit_transform(feature_test_float) 

print feature_train_rescaled,feature_test_rescaled
#t0=time.time()
#clf = Perceptron(n_iter=1000)
#clf.fit(feature_train_float,label_train)
#print "Training time ",round(time.time()-t0,3),"s"
#print "Number of Classes :",len (clf.classes_)

#label_pred = clf.fit(feature_train_float,label_train).predict(feature_test_float)

# Step 4 : Generate Confusion Matrix

#confusion_matrix = confusion_matrix(label_test,label_pred)
#print confusion_matrix

# Step 5 : Accuracy 

#score = clf.score(feature_test_float,label_test)
#print "Test Accuracy :",score 


