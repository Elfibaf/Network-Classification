# Authors : Mehdi Crozes and Fabien Robin
# Date : June 10th 2016

# LogisticClassifier for ARFF traffic network file 

import arff
import numpy as np
import matplotlib.pyplot as py
import time


from extraction import *
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split,KFold
from sklearn.metrics import precision_score,recall_score

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")
print "\tTotal dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = split_data(arff_file)

#feature_train,feature_test,label_train,label_test = kfold_data(arff_file)

# Step 3 : Modeling and training our classifier with training's time
#Parameters for LogisticRegression : solver to minimize loss function and multi_class='multinomial' for multi_class problem to use the sofmax function so as to find the predicted probability of each class


clf = LogisticRegression(penalty='l2',max_iter=5,solver='lbfgs',multi_class='ovr')
t0 = time.time()
clf.fit(feature_train,label_train)

print "\tTraining time: ",round(time.time()-t0,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

t1 = time.time()
label_pred = clf.predict(feature_test)
print "\tPredicting time: ",round(time.time()-t1,3),"s"

# Step 4 : Generate precision_score and recall_score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# S1tep 5 : Accuracy 

score = clf.score(feature_test,label_test)
score2 = clf.score(feature_train,label_train)
print "\tTest Accuracy :",score 
print "\tTraining Accuracy :",score2

