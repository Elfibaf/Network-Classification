# Authors : Fabien Robin and Mehdi Crozes
# Date : June 8th 2016

# K Nearest Neighbours classifier for ARFF traffic network file 

import re
import numpy as np
import os
import arff
import extraction

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split,KFold
from sklearn.grid_search import GridSearchCV


data = extraction.load_dataset("../../Data/Caida/features_stats_best.arff")
print "\tTotal dataset : "
print "\tNumber of samples:",data.nb_examples
print "\tNumber of features:",len(data.features[0])

# Splitting data into a training and testing set

#feature_train,feature_test,label_train,label_test = train_test_split(data.features,data.labels,test_size=0.25,random_state=42)

feature_train,feature_test,label_train,label_test = extraction.kfold_data(data,10)

"""param_grid = [
  {'leaf_size': [30, 20, 10, 40, 50], 'n_neighbors' : [5, 6, 7]},
 ]"""


#model = GridSearchCV(KNeighborsClassifier(), param_grid)
model = KNeighborsClassifier()
model.fit(feature_train, label_train)
print(len(model.classes_))

predicted = model.predict(feature_test)

print("Prediction : ", predicted, "\n")
print("Labels : ", label_test, "\n")
print("Score : ", model.score(feature_test, label_test))
