# Authors : Mehdi Crozes and Fabien Robin
# Date : June 7th 2016

# GaussianNB for ARFF traffic network file 

import arff
import numpy as np
import time
import math

from extraction import * 
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_score
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_classif
from sklearn.preprocessing import MinMaxScaler

# Step 1 : Import Arff file : right here it's just 6 features

arff_file = load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")
print "Total dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Fitting and training our classifier 
# Before that, we had to convert our feature_train_float into float type because our feature_train is an array of string and generate a TypeError

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

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

