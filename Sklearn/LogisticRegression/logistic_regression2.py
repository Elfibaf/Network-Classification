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
from sklearn.metrics import confusion_matrix,precision_score,recall_score

# Step 1 : Import Arff file with only 6 features

arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")
print "\tTotal dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Modeling and training our classifier with training's time

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

#Parameters for LogisticRegression : solver to minimize loss function and multi_class='multinomial' for multi_class problem to use the sofmax function so as to find the predicted probability of each class


clf = LogisticRegression(penalty='l2',max_iter=20,solver='newton-cg',multi_class='ovr')
t0 = time.time()
clf.fit(feature_train_float,label_train)

print "\tTraining time: ",round(time.time()-t0,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

t1 = time.time()
label_pred = clf.predict(feature_test_float)
print "\tPredicting time: ",round(time.time()-t1,3),"s"

# Step 4 : Generate precision_score and recall_score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# S1tep 5 : Accuracy 

score = clf.score(feature_test_float,label_test)
score2 = clf.score(feature_train_float,label_train)
print "\tTest Accuracy :",score 
print "\tTraining Accuracy :",score2

