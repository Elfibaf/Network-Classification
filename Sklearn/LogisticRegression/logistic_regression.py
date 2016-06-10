# Authors : Mehdi Crozes and Fabien Robin
# Date : June 10th 2016

# LogisticClassifier for ARFF traffic network file 

import arff
import numpy as np
import matplotlib.pyplot as py
import time

from extraction import *
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Modeling and training our classifier with training's time

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

t0 = time.time()
clf = LogisticRegression(penalty='l2',solver='newton-cg',max_iter=50,multi_class='multinomial')
clf.fit(feature_train_float,label_train)

print "Training time: ",round(time.time()-t0,3),"s"
print "Number of Classes :",len (clf.classes_)

t1 = time.time()
clf.predict(feature_test_float)
print "Predicting time: ",round(time.time()-t1,3),"s"

# Step 4 : Generate Confusion Matrix

#confusion_matrix = confusion_matrix(label_test,label_pred)
#print confusion_matrix

# Step 5 : Accuracy 

score = clf.score(feature_test_float,label_test)
score2 =clf.score(feature_train_float,label_train)
print "Test Accuracy :",score 
print "Training Accuracy :",score2

