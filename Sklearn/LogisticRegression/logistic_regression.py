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
from sklearn.grid_search import GridSearchCV

# Step 1 : Import Arff file

arff_file = load_dataset("../../Data/data_caida_original.arff")

# Step 2 : Building training_set and test_set

feature_train,feature_test,label_train,label_test = train_test_split(arff_file.features,arff_file.labels,test_size=0.25,random_state=42)

# Step 3 : Modeling and training our classifier with training's time

feature_train_float = feature_train.astype(np.float)
feature_test_float = feature_test.astype(np.float)

#Parameters for LogisticRegression : solver to minimize loss function and multi_class='multinomial' for multi_class problem to use the sofmax function so as to find the predicted probability of each class

tuned_parameters=[{'C':[1,10,100,1000],'max_iter':[5,10,15,20,30],'multi_class':('ovr','multinomial')}]

scores ='precision'

clf = GridSearchCV(LogisticRegression(penalty='l2',C=1,max_iter=5,solver='lbfgs',multi_class='ovr'),tuned_parameters,cv=10,scoring=scores)
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

