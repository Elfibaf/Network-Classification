# Authors : Fabien Robin and Mehdi Crozes
# Date : June 9th 2016

# SVM classifier for ARFF traffic network file 

import re
import numpy as np
import os
import arff
import extraction2

from sklearn import svm
from sklearn.cross_validation import train_test_split,KFold
from sklearn.preprocessing import normalize
from sklearn.grid_search import GridSearchCV

data = extraction2.load_dataset("../../Data/Caida/features_caida_flowcalc2.arff")

print "\tTotal dataset : "
print "\tNumber of samples:",data.nb_examples
print "\tNumber of features:",len(data.features[0])

#data.features = normalize(data.features)
#print(data.features)

feature_train,feature_test,label_train,label_test = split_data(arff_file)

#feature_train,feature_test,label_train,label_test = kfold_data(arff_file,10)

# Searching for best parameters for the classifier
param_grid = [
  {'C': [1, 10, 100, 1000], 'gamma': np.logspace(-20,0,num=21), 'kernel': ['rbf']},
 ]

model = GridSearchCV(svm.SVC(C=1), param_grid)

X = feature_train
Y = label_train

model.fit(X, Y)

print("Best parameters : ", model.best_params_)

print(model.score(X, Y))

predicted = model.predict(feature_test)
print(model.score(feature_test, label_test))
print(predicted)
