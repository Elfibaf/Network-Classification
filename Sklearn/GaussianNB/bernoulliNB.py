# Authors : Mehdi Crozes and Fabien Robin
# Date : June 7th 2016
# Update : June 22nd 2016

# BernouilliNB with feature selection Weka  for ARFF traffic network file 

import arff
import numpy as np
import time
import math

from extraction import * 
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split,KFold
from sklearn.metrics import recall_score,precision_score
from sklearn.preprocessing import MinMaxScaler

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")
print "Total dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = split_data(arff_file)

#feature_train,feature_test,label_train,label_test = kfold_data(arff_file)

# Step 3 : Fitting and training our classifier 

scaler = MinMaxScaler(copy='false')
feature_train_rescaled = scaler.fit_transform(feature_train_float)
feature_test_rescaled = scaler.fit_transform(feature_test_float) 

clf = BernoulliNB(binarize=0.01)
t0 = time.time()
clf.fit(feature_train_rescaled,label_train)
print "\tTraining time ",round(time.time()-t0,3),"s"

log_prob_class = clf.class_log_prior_
prob_class = [math.pow(10,log_prob_class[i]) for i in log_prob_class]
print "\tMax probablilite ",round(max(prob_class),10)
	

t1 = time.time()
label_pred = clf.predict(feature_test_rescaled)
print "\tPredicting time: ",round(time.time()-t1,3),"s"

# Step 4 : Precision_score and recall_score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')

# Step 5 : Accuracy 

score = clf.score(feature_test_rescaled,label_test)
score2 = clf.score(feature_train_rescaled,label_train)
print "\tTest Accuracy :",score
print "\tTrain Accuracy:",score2

