# Authors : Fabien Robin and Mehdi Crozes
# Date : June 9th 2016

# SVM classifier for ARFF traffic network file 

import re
import numpy as np
import os
import arff
from extraction2 import *

from sklearn import svm
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import normalize


arff_file = load_dataset_barcelona("../../Data/Info_file/packets_all_1.info")
print "\tTotal dataset : "
print "\tNumber of samples:",arff_file.nb_examples
print "\tNumber of features:",len(arff_file.features[0])

#feature_train,feature_test,label_train,label_test = split_data(arff_file)

feature_train,feature_test,label_train,label_test = kfold_data(arff_file,10)

model = svm.SVC(C=1,random_state=1)
X = feature_train
Y = label_train

model.fit(X, Y)
print(model.score(X, Y))

predicted = model.predict(feature_test)
print(model.score(feature_test, label_test))
print(predicted)
