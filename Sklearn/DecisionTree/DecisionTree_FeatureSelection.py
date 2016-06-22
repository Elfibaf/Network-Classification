# Authors : Mehdi Crozes and Fabien Robin
# Date : June 7th 2016

# DecisionTreeClassifier for ARFF traffic network file 

import arff
import numpy as np
import matplotlib.pyplot as py
import time

from extraction import *
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix,recall_score,precision_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

# Step 1 : Import Arff file

arff_file = load_dataset("../../../Data/Caida/Tests_features/All/features_all_time.arff")
print "\tTotal dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

# Step 2 : Building training_set and test_set


selector = SelectKBest(f_classif, k=4)
features = arff_file.features.astype(np.float)
new_feat = selector.fit_transform(features, arff_file.labels)
feature_train_float,feature_test_float,label_train,label_test = train_test_split(new_feat,arff_file.labels,test_size=0.25,random_state=42)

print("Best scores : ")
print(selector.scores_)

# Step 3 : Fitting and training our classifier 

#feature_train_float = feature_train.astype(np.float)
#feature_test_float = feature_test.astype(np.float)

print(feature_train_float[1])

clf = tree.DecisionTreeClassifier()

t0 = time.time()
clf.fit(feature_train_float,label_train)

print "\tTraining time: ",round(time.time()-t0,3),"s"
print "\tNumber of Classes :",len (clf.classes_)

t1 = time.time()
label_pred = clf.predict(feature_test_float)
print "\tPredicting time: ",round(time.time()-t1,3),"s"

# Step 4 : Precision_score and recall_score

print "\tPrecision :",precision_score(label_test,label_pred,average='micro')
print "\tRecall :",recall_score(label_test,label_pred,average='micro')


# Step 5 : Accuracy 

score = clf.score(feature_test_float,label_test)
score2 = clf.score(feature_train_float,label_train)
print "\tTest Accuracy :",score 
print "\tTeain Accuracy:",score2
